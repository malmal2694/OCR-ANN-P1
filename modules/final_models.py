from torch.nn import Module, Sequential
from . import decoder, encoder

class LineRecognition(Module):
    def __init__(self):
        super().__init__()
        
        self.model = Sequential(encoder.FCN_Encoder, decoder.LineDecoder)
    
    def forward(self, line_img):
        return self.model(line_img)