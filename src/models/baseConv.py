""" A base class for Convolution. """
import torch.nn as nn


class BaseConv(nn.Module):
    """
    Applies a multi-layer 1d convolution to an input sequence.
    Note:
        Do not use this class directly, use one of the sub classes.
    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): maximum allowed length for the sequence to be processed
        hidden_size (int): number of features in the hidden state `h`
        input_dropout_p (float): dropout probability for the input sequence
        dropout_p (float): dropout probability for the output sequence
        n_layers (int): number of recurrent layers
        rnn_cell (str): type of RNN cell (Eg. 'LSTM' , 'GRU')
    Inputs: ``*args``, ``**kwargs``
        - ``*args``: variable length argument list.
        - ``**kwargs``: arbitrary keyword arguments.
    Attributes:
        SYM_MASK: masking symbol
        SYM_EOS: end-of-sequence symbol
    """
    SYM_MASK = "MASK"
    SYM_EOS = "EOS"

    def __init__(self, vocab_size, kernel_size, hidden_size, n_channels,
                 n_layers, input_dropout_p, dropout_p, max_len_blocks):
        super(BaseConv, self).__init__()
        self.vocab_size = vocab_size
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size
        self.n_channels = n_channels
        self.n_layers = n_layers
        self.input_dropout_p = input_dropout_p
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        self.dropout_p = dropout_p
        self.max_len_blocks = max_len_blocks
        self.conv_method = nn.Conv1d

    def forward(self, *args, **kwargs):
        raise NotImplementedError()