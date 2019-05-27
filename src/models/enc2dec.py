import torch.nn.functional as F

from .baseModel import BaseModel

class Enc2dec(BaseModel):
    """ Standard encoder-to-decoder architecture with configurable encoder
    and decoder.
    """

    def __init__(self, encoder, decoder, decode_function=F.log_softmax, which_enc="rnn", which_dec="rnn"):
        super(Enc2dec, self).__init__(encoder_module=encoder,
                                      decoder_module=decoder, decode_function=decode_function)

        self.which_enc = which_enc
        self.which_dec = which_dec

    # def flatten_parameters(self):
    #     """
    #     Flatten parameters of all components in the model.
    #     """
    #     self.encoder_module.rnn.flatten_parameters()
    #     self.decoder_module.rnn.flatten_parameters()

    def forward(self, inputs, block_inputs, input_lengths=None,
                ins_position_variables=None, position_variables=None):
        # Unpack target variables

        hidden = None
        if self.which_enc == "conv":
            output = self.encoder_module(inputs, ins_position_variables)
        else:
            output, hidden = self.encoder_module(inputs, input_lengths=input_lengths)
        if self.which_dec == "conv":
            predicted_softmax, attn = self.decoder_module(block_inputs, position_variables, output, F.log_softmax)
        else:
            if self.which_enc == "conv":
                hidden = None
            predicted_softmax, hidden, attn = self.decoder_module(block_inputs, hidden, output, F.log_softmax)
        return predicted_softmax, attn