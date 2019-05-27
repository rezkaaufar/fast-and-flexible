import torch.nn as nn
import torch.nn.functional as F

import sys
import abc
import types

if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(types.StringType('ABC'), (), {})


class BaseModel(ABC, nn.Module):
    """
    Abstract base class for models.
    Args:
        encoder_module (baseRNN or baseConv) :  module that encodes inputs
        decoder_module (baseRNN or baseConv, optional):   module to decode encoded inputs
        decode_function (callable, optional): function to generate symbols from output hidden states (default: F.log_softmax)
    """

    def __init__(self, encoder_module, decoder_module=None,
                 decode_function=F.log_softmax):
        super(BaseModel, self).__init__()
        self.encoder_module = encoder_module
        self.decoder_module = decoder_module
        self.decode_function = decode_function

    # def flatten_parameters(self):
    #     """
    #     Flatten parameters of all components in the model.
    #     """
    #     raise NotImplementedError(
    #         "A generic version of this function should be implemented")

    def reset_parameters(self):
        """
        Reset the parameters of all components in the model.
        """
        raise NotImplementedError(
            "A generic version of this function should be implemented")

    def forward(self, inputs, block_inputs, input_lengths=None):
        """
        :param inputs:
        :param block_inputs:
        :param input_lengths:
        :return:
        """