import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import Attention
from .baseRNN import BaseRNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DecoderRNN(BaseRNN):
    """
    Provides functionality for decoding with a sequential RNN module and an option for attention.
    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        n_layers (int, optional): number of recurrent layers (default: 1)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        bidirectional (bool, optional): if the encoder is bidirectional (default False)
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        use_attention(bool, optional): flag indication whether to use attention mechanism or not (default: false)
        full_focus(bool, optional): flag indication whether to use full attention mechanism or not (default: false)
    Attributes:
        KEY_ATTN_SCORE (str): key used to indicate attention weights in `ret_dict`
        KEY_LENGTH (str): key used to indicate a list representing lengths of output sequences in `ret_dict`
        KEY_SEQUENCE (str): key used to indicate a list of sequences in `ret_dict`
    Inputs: inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio
        - **inputs** (batch, seq_len, input_size): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs.  It is used for teacher forcing when provided. (default `None`)
        - **encoder_hidden** (num_layers * num_directions, batch_size, hidden_size): tensor containing the features in the
          hidden state `h` of encoder. Used as the initial hidden state of the decoder. (default `None`)
        - **encoder_outputs** (batch, seq_len, hidden_size): tensor with containing the outputs of the encoder.
          Used for attention mechanism (default is `None`).
        - **function** (torch.nn.Module): A function used to generate symbols from RNN hidden state
          (default is `torch.nn.functional.log_softmax`).
        - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used. A random number is
          drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0).
    Outputs: decoder_outputs, decoder_hidden, ret_dict
        - **decoder_outputs** (seq_len, batch, vocab_size): list of tensors with size (batch_size, vocab_size) containing
          the outputs of the decoding function.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
          representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
          predicted token IDs }.
    """

    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'

    def __init__(self, vocab_size, max_len, hidden_size,
                 n_layers=1, rnn_cell='gru', bidirectional=False,
                 input_dropout_p=0, dropout_p=0, use_attention=False, attention_method=None, full_focus=False):
        super(DecoderRNN, self).__init__(vocab_size, max_len, hidden_size,
                                         input_dropout_p, dropout_p,
                                         n_layers, rnn_cell)

        self.bidirectional_encoder = bidirectional
        input_size = hidden_size

        if use_attention and attention_method is None:
            raise ValueError(
                "Method for computing attention should be provided")

        self.attention_method = attention_method
        self.full_focus = full_focus

        # increase input size decoder if attention is applied before decoder
        # rnn
        if use_attention == 'pre-rnn' and not full_focus:
            input_size *= 2

        self.rnn = self.rnn_cell(input_size, hidden_size, n_layers,
                                 batch_first=True, dropout=dropout_p)

        self.output_size = vocab_size
        self.max_length = max_len
        self.use_attention = use_attention

        self.init_input = None

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        if use_attention:
            self.attention = Attention(self.hidden_size, self.attention_method)
        else:
            self.attention = None

        if use_attention == 'post-rnn':
            self.out = nn.Linear(2 * self.hidden_size, self.output_size)
        else:
            self.out = nn.Linear(self.hidden_size, self.output_size)
            if self.full_focus:
                self.ffocus_merge = nn.Linear(
                    2 * self.hidden_size, hidden_size)

    def forward(self, input_var, hidden,
                     encoder_outputs, function, **kwargs):
        """
        Performs one or multiple forward decoder steps.
        Args:
            input_var (torch.tensor): Variable containing the input(s) to the decoder RNN
            hidden (torch.tensor): Variable containing the previous decoder hidden state.
            encoder_outputs (torch.tensor): Variable containing the target outputs of the decoder RNN
            function (torch.tensor): Activation function over the last output of the decoder RNN at every time step.
        Returns:
            predicted_softmax: The output softmax distribution at every time step of the decoder RNN
            hidden: The hidden state at every time step of the decoder RNN
            attn: The attention distribution at every time step of the decoder RNN
        """
        batch_size = input_var.size(0)
        output_size = input_var.size(1)
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)
        output = None
        hidden = self._init_state(hidden)

        if self.use_attention == 'pre-rnn':
            h = embedded
            # Apply the attention method to get the attention vector and weighted context vector. Provide decoder step for hard attention
            # transpose to get batch at the second index
            #print(h.size(), encoder_outputs.size())
            context, attn = self.attention(h, encoder_outputs, **kwargs)
            #print(context.size(), attn.size(), embedded.size())
            #context = context.repeat(1, embedded.size(1), 1)
            combined_input = torch.cat((context, embedded), dim=2)
            if self.full_focus:
                merged_input = F.relu(self.ffocus_merge(combined_input))
                combined_input = torch.mul(context, merged_input)
            output, hidden = self.rnn(combined_input, hidden)

        elif self.use_attention == 'post-rnn':
            output, hidden = self.rnn(embedded, hidden)
            # Apply the attention method to get the attention vector and
            # weighted context vector. Provide decoder step for hard attention
            context, attn = self.attention(output, encoder_outputs, **kwargs)
            output = torch.cat((context, output), dim=2)

        elif not self.use_attention:
            attn = None
            output, hidden = self.rnn(embedded, hidden)

        predicted_softmax = function(self.out(
            output.contiguous().view(-1, self.out.in_features)), dim=1).view(batch_size, output_size, -1)

        return predicted_softmax, hidden, attn

    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h)
                                    for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h