from .dataset import OCRDataset, dataloader_collate_fn
import torch.optim as optim
from torch.nn import CTCLoss
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, Normalize


class TrainModel:
    def __init__(self, model, dataset, model_params, dataset_params):
        """
        model: Name of model to train
        """
        self.model = model(model_params)
        dataset = dataset(dataset_params)
        self.dataloader = DataLoader(
            dataset, batch_size=2, shuffle=True, collate_fn=dataloader_collate_fn
        )

    def fit(self, opt_lr, num_epoch=2):
        optimizer = optim.Adam(self.model.parameters(), lr=opt_lr)
        loss_fn = CTCLoss(reduction="sum")

        for epoch in range(num_epoch):
            running_loss = 0.0
            for i, data in enumerate(self.dataloader):
                imgs = data["img"]
                gts = data["gt"]

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(imgs)
                loss = loss_fn(
                    outputs,
                    gts,
                    [img.shape[1] for img in imgs],
                    [len(gt) for gt in gts],
                )
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                # if i % 5 == 0:
                if i % 1 == 0:
                    print(
                        f"Iteration {i+1} of epoch {epoch}) loss: {running_loss / 5:.4f} "
                    )
                    running_loss = 0
