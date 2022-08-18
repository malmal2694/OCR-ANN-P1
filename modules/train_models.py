from .dataset import dataloader_collate_fn
import torch.optim as optim
# from torch.nn import CTCLoss
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F

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

    def fit(self, opt_lr, num_epoch=2, debug_mode=False):
        """
        Train the model
        
        Parameters
        ----------
        debug_mode: If true, show more information in training
        """
        optimizer = optim.Adam(self.model.parameters(), lr=opt_lr)
        # loss_fn = CTCLoss(blank=0, reduction="sum", zero_infinity=True)
        loss_fn = CTCLoss(blank=0)
        if debug_mode: torch.autograd.set_detect_anomaly(True)
            
        for epoch in range(num_epoch):
            running_loss = 0.0
            for i, data in enumerate(self.dataloader):
                # Send image and gt batch to the device that is specified.
                imgs = data["img"].to(self.device)
                gts = data["gt"].to(self.device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                output = self.model(imgs)
                if debug_mode:
                    print(f"Shape of the output of the model: {output.shape}")
                    print(f"imgs.shape: {imgs.shape}")
                    print(f"gts.shape: {gts.shape}")
                
                # loss = loss_fn(
                #     output.permute(2, 0, 1),
                #     gts,
                #     torch.tensor(imgs.size(0) * [output.size(0)]),
                #     torch.tensor([gts.size(0) * [gts.size(1)]]),
                # )
                loss = loss_fn(output, gts)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 5 == 0:
                    print(f"Iteration {i} of epoch {epoch}) loss: {(running_loss / 5):.5f}")
                    running_loss = 0

class CTCLoss(nn.Module):
    """
    Convenient wrapper for CTCLoss that handles log_softmax and taking 
    input/target lengths.
    Source code: https://discuss.pytorch.org/t/best-practices-to-solve-nan-ctc-loss/151913
    """

    def __init__(self, blank: int = 0) -> None:
        """
        Init method.

        Parameters
        ----------
        blank (int, optional): Blank token. Defaults to 0.
        """
        super().__init__()
        self.blank = blank

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward method.

        Parameters
        ----------
        preds (torch.Tensor): Model predictions. Tensor of shape (batch, sequence_length, num_classes), or (N, T, C).
        targets (torch.Tensor): Target tensor of shape (batch, max_seq_length). max_seq_length may vary
        per batch.

        Returns
        -------
        torch.Tensor: Loss scalar.
        """
        preds = preds.log_softmax(-1)
        batch, seq_len, classes = preds.shape
        # preds = preds.permute(1, 0, 2) # since ctc_loss needs (T, N, C) inputs
        preds = preds.permute(2, 0, 1)
        pred_lengths = torch.full(size=(batch,), fill_value=seq_len, dtype=torch.long)
        target_lengths = torch.count_nonzero(targets, axis=1)

        return F.ctc_loss(preds, targets, pred_lengths, target_lengths, blank=self.blank, zero_infinity=True)