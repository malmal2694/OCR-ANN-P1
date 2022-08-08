from torch.nn import Module, Sequential
from . import decoder, encoder
from . import params

class LineRecognition(Module):
    def __init__(self):
        super().__init__()
        self.model = Sequential(encoder.FCN_Encoder(params.params), decoder.LineDecoder(params.params))
    
    def forward(self, sample):
        return self.model(sample)
