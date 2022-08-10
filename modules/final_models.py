from torch.nn import Module, Sequential
from . import decoder, encoder
from . import params


class LineRecognition(Module):
    def __init__(self, params):
        super().__init__()
        self.model = Sequential(
            encoder.FCN_Encoder(params), decoder.LineDecoder(params)
        )

    def forward(self, sample):
        return self.model(sample)
