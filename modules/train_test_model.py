from .dataset import dataloader_collate_fn, Normalize
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torch import nn
from os import path
from pathlib import Path
from .utils import DecodeString


class TrainModel:
    def __init__(self, model, dataset, params:dict, show_log_steps:int, save_check_step:int, lr=None) -> None:
        """
        model: Name of model to train
        
        Parameters
        ----------
        show_log_step (int): Show log of the model at each "show_log_step" dataloader iteration
        save_check_step (int): Save a new checkpoint at each "save_check_Step" epoch
        lr (int): If it doesn't set, use the learning rate specified in the parameters dictionary
        """
        self.device = params["training_params"]["device"]
        self.model = model(params).to(self.device)
        self.dataset = dataset(params)
        self.params = params
        self.dataloader = DataLoader(
            self.dataset,
            batch_size = params["training_params"]["batch_size"],
            shuffle=True,
            collate_fn=dataloader_collate_fn,
        )
        self.show_log_step = show_log_steps
        self.loss_fn = CTCLoss(blank=0)
        # The last epoch executed in the last run
        self.last_epoch_index = 0
        self.lr = self.params["training_params"]["min_opt_lr"] if lr == None else lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.max_epoch = self.params["training_params"]["epoch_numbers"]
        self.checkpoint_dir = self.params["training_params"]["checkpoint_dir"]
        self.save_check_step = save_check_step
        self.test_model = TestModel(self.model, self.device, self.params["unique_chars_map_file"])
        
    def fit(self, debug_mode=False):
        """
        Train the model
        
        Parameters
        ----------
        debug_mode: If true, show more information in training
        """
        torch.autograd.set_detect_anomaly(True)
            
        for epoch in range(self.last_epoch_index, self.max_epoch):
            running_loss = 0.0
            for i, data in enumerate(self.dataloader):
                # Send image and gt batch to the device that is specified.
                imgs = data["img"].to(self.device)
                gts = data["gt"].to(self.device)
                
                # zero the parameter gradients
                self.optimizer.zero_grad()
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
                loss = self.loss_fn(output, gts)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % self.show_log_step == 0:
                    print(f"Iteration {i} of epoch {epoch}) loss: {(running_loss / self.show_log_step):.5f}")
                    running_loss = 0
                    # Create a test set and a prediction for this set
                    test_set = self.dataset[0]
                    self.test_model(test_set["gt"], test_set["img"].unsqueeze(0))
                    
            if epoch % self.save_check_step == 0:
                out = self.save_checkpoint(epoch)
                print(f"Iteration {i} of epoch {epoch}) Checkpoint saved. checkpoint path: {out}")
                
    def load_checkpoint(self, file_name:str) -> None:
        """
        Load the checkpoint.
        Checkpoints contain, parameters of the model, optimizer, loss value, and index of the last epoch.
        
        Parameters
        ----------
        file_name (str): Name of the file. (e.g., file-name.pt)
        """
        checkpoint = torch.load(path.join(self.checkpoint_dir, file_name), map_location=self.device)
        self.last_epoch_index = checkpoint["last_epoch_index"] + 1
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss_fn.load_state_dict(checkpoint["loss_state_dict"])

    def save_checkpoint(self, index:int) -> str:
        """
        Parameters
        ----------
        file_name (str): Name of file to store in the dir_path directory. 
        (e.g., file-name not file-name-cp-10.pt)(cp: check point)
        
        Returns
        -------
        Path and name of the file that created
        """
        file_name = self.params["training_params"]["checkpoint_name"]
        hash_count = file_name.count("#")
        file_name = file_name.replace(hash_count * "#", format(index, f"0{hash_count}d"))
        file_path = path.join(self.checkpoint_dir, file_name)
        # If file with the same name exist, throw an error
        if Path(file_path).is_file():
            raise FileExistsError(f"A file  with the same name and path exist.\nFile name: {file_path}")
        torch.save({
            "last_epoch_index": index,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss_state_dict": self.loss_fn.state_dict()
        }, file_path)
        
        return file_path

class TestModel:
    """
    Test the given model
    """
    def __init__(self, params:dict, model, device, map_char_file:str):
        """
        Parameters
        ----------
        params (dict): A dictionary contains of the parameters
        model: A object of the model not a class of the model
        device: The device the model is on
        map_char_file (str): Address of the file maps int to char
        """
        self.device = params["training_params"]["device"]
        self.model = model().to(device)
        self.decode_string = DecodeString(map_char_file)
        self.params = params
        self.normalizer = Normalize(False)
        
    @torch.no_grad()
    def __call__(self, img:torch.Tensor) -> str:
        """
        Parameters
        ----------
        img (torch.tensor): Image to convert to text. Note that pixel's value 
        are in the range of 0-255.
        """
        # Predicted gt returned from the model
        img = self.normalizer(img)
        pred_gt = self.model(img.to(self.device))
        
        # Return characters with highest probability.
        # An example is below:
        # pred_gt =  torch.tensor([
        #     [[1,2,3]],
        #     [[4,50,6]],
        #     [[1,6,3]],
        #     [[5,8,4]]
        # ])
        # pred_gt.max(0):
        # torch.return_types.max(
        # values=tensor([[ 5, 50,  6]]),
        # indices=tensor([[3, 1, 1]]))
        # pred_gt = pred_gt.max(1).indices.flatten()
        
        # pred_gt = self.decode_string(pred_gt)
        return pred_gt

    def load_checkpoint(self, checkpoint_path:str) -> None:
        """
        Load the checkpoint.
        Checkpoints contain parameters of the model, optimizer, loss value, and index of the last epoch.
        
        Parameters
        ----------
        checkpoint_path (str): path of the checkpoint. (e.g., checkpoint/path/file-name.pt)
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

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
        # preds = preds.log_softmax(-1)
        # batch, seq_len, classes = preds.shape
        batch, classes, seq_len = preds.shape
        # preds = preds.permute(1, 0, 2) # since ctc_loss needs (T, N, C) inputs
        preds = preds.permute(2, 0, 1)
        pred_lengths = torch.full(size=(batch,), fill_value=seq_len, dtype=torch.long)
        target_lengths = torch.count_nonzero(targets, axis=1)

        return F.ctc_loss(preds, targets, pred_lengths, target_lengths, blank=self.blank, zero_infinity=True)
    
