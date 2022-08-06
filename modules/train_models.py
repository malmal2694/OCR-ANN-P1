from . import decoder, encoder, params
from . import final_models as models
import torch.optim as optim


class LineModelTrainer():
    def fit(self):
        opt_lr = params.params["training_params"]["opt_lr"]
        opt = optim.Adam(models.LineRecognition.parameters(), lr=opt_lr)
        while():
            for index, batch in enumerate(iter(data)):
                pass