from torch.nn import Module, Sequential
from . import decoder, encoder


class LineRecognition(Module):
    def __init__(self, params):
        super().__init__()
        self.model = Sequential(
            encoder.FCN_Encoder(params), decoder.LineDecoder(params)
        )

    def forward(self, sample):
        """
        Forward the images not the image/gt pairs
        Return predicted gt for given img
        """
        return self.model(sample)
