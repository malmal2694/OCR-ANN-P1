from .dataset import dataloader_collate_fn
import torch.optim as optim
from torch.nn import CTCLoss
from torch.utils.data import DataLoader
import torch


class TrainModel:
    def __init__(self, model, dataset, model_params, dataset_params):
        """
        model: Name of model to train
        """
        self.device = model_params["training_params"]["device"]
        self.model = model(model_params).to(self.device)
        dataset = dataset(dataset_params)
        self.model_params = model_params
        self.dataloader = DataLoader(
            dataset,
            batch_size=model_params["training_params"]["batch_size"],
            shuffle=True,
            collate_fn=dataloader_collate_fn,
        )

    def fit(self, opt_lr, num_epoch=2):
        optimizer = optim.Adam(self.model.parameters(), lr=opt_lr)
        loss_fn = CTCLoss(reduction="sum")

        for epoch in range(num_epoch):
            running_loss = 0.0
            for i, data in enumerate(self.dataloader):
                # Send image and gt batch to the device that is specified.
                imgs = data["img"].to(self.device)
                gts = data["gt"].to(self.device)
                # zero the parameter gradients
                optimizer.zero_grad()
                print(f"(train_models.py)Image batch shape: {imgs.shape}")

                # forward + backward + optimize
                output = self.model(imgs)
                loss = loss_fn(
                    output.permute(2, 0, 1),
                    gts,
                    torch.tensor(imgs.size(0) * [output.size(0)]),
                    torch.tensor([gts.size(0) * [gts.size(1)]]),
                )
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                # if i % 5 == 0:
                if i % 1 == 0:
                    print(f"Iteration {i} of epoch {epoch}) loss: {running_loss:.5f}")
                    running_loss = 0
