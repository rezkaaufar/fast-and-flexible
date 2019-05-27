import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import Attention
from .baseConv import BaseConv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderConv(BaseConv):
    """
    Provides functionality for decoding with a convolutional module and an option for attention.
    Args:
        vocab_size (int): size of the vocabulary
        kernel_size (int): size of the convolution kernel
        hidden_size (int): the number of features in the hidden state `h`
        n_channels (int): size of the convolution channels
        n_layers (int, optional): number of convolutional layers (default: 1)
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        max_len_blocks (int, optional): maximum number of blocks length for positional embeddings (default: 23)
    Inputs: inputs, position_inputs, encoder_outputs, function
        - **inputs** (batch, seq_len, input_size): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs.  It is used for teacher forcing when provided. (default `None`)
        - **position_inputs** (batch_size, seq_len, input_size): list of sequences containing the IDs of the token
          positions. (default `None`)
        - **encoder_outputs** (batch, seq_len, hidden_size): tensor with containing the outputs of the encoder.
          Used for attention mechanism (default is `None`).
        - **function** (torch.nn.Module): A function used to generate symbols from Convolution hidden state
          (default is `torch.nn.functional.log_softmax`).
    Outputs: decoder_outputs, decoder_hidden, ret_dict
        - **decoder_outputs** (seq_len, batch, vocab_size): list of tensors with size (batch_size, vocab_size) containing
          the outputs of the decoding function.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
          representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
          predicted token IDs }.
    """

    def __init__(self, vocab_size, kernel_size, hidden_size, n_channels,
                 n_layers=1, input_dropout_p=0, dropout_p=0, max_len_blocks=23):
        super(EncoderConv, self).__init__(vocab_size, kernel_size, hidden_size, n_channels, n_layers,
                                          input_dropout_p, dropout_p, max_len_blocks)

        input_size = n_channels
        self.output_size = vocab_size

        self.conv = nn.ModuleList([self.conv_method(input_size, input_size, kernel_size,
                                                    padding=kernel_size // 2) for _ in range(n_layers)])

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.position_embedding = nn.Embedding(self.max_len_blocks, self.hidden_size)

    def forward(self, input_var, input_pos, **kwargs):
        """
        Performs one or multiple forward decoder steps.
        Args:
            input_var (torch.tensor): Variable containing the input(s) to the decoder Convolution
            input_pos (torch.tensor): Variable containing the position input(s) to the decoder Convolution
        Returns:
            predicted_softmax: The output softmax distribution at every time step of the decoder Convolution
        """
        embedded = self.embedding(input_var)
        position_embedded = self.position_embedding(input_pos)
        embedded = self.input_dropout(embedded + position_embedded)

        h = embedded

        h = h.transpose(1, 2)
        for i, layer in enumerate(self.conv):
            h = F.tanh(layer(h) + h)
        h = h.transpose(1, 2)

        output = h

        return output