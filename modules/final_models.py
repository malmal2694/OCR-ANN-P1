from torch.nn import Module, Sequential
from .line_model import l_decoder, l_encoder


class LineRecognition(Module):
    def __init__(self, params):
        super().__init__()
        self.model = Sequential(
            l_encoder.FCN_Encoder(params), l_decoder.LineDecoder(params)
        )

    def forward(self, sample):
        """
        Forward the images not the image/gt pairs
        """
        return self.model(sample)
