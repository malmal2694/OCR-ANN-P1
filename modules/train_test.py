from .dataset import dataloader_collate_fn, Normalize
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torch import nn
from os import path
from pathlib import Path
from .utils import DecodeString, load_char_map_file, CodingString, clean_sentence
from fast_ctc_decode import viterbi_search
import numpy as np
from .statistic import word_error_rate, char_error_rate
from typing import Tuple
from torch.optim.lr_scheduler import ReduceLROnPlateau


class TrainModel:
    """
    Train the model.
    """

    def __init__(
        self, model, dataset, params: dict, show_log_steps: int, save_check_step: int
    ) -> None:
        """
        model: Name of model to train

        Parameters
        ----------
        show_log_step (int): Show log of the model at each "show_log_step" dataloader iteration
        save_check_step (int): Save a new checkpoint at each "save_check_Step" epoch
        lr (int): If it doesn't set, use the learning rate specified in the parameters dictionary
        """
        self.device = params["training"]["device"]
        self.model = model(params).to(self.device)
        self.dataset = dataset(params)
        self.params = params
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=params["training"]["batch_size"],
            shuffle=True,
            collate_fn=dataloader_collate_fn,
        )
        self.show_log_step = show_log_steps
        self.loss_fn = CTCLoss(blank=0)
        # The last epoch executed in the last run
        self.last_epoch_index = 0
        self.start_lr_val = self.params["training"]["lr"]["start_lr"]
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.start_lr_val)
        self.max_epoch = self.params["training"]["epoch_numbers"]
        self.checkpoint_dir = self.params["training"]["checkpoint_dir"]
        self.save_check_step = save_check_step
        self.map_char_file = params["dataset"]["unique_chars_map_file"]
        self.alphabet = "".join(load_char_map_file(self.map_char_file).keys())
        # Add a character is not used in the alphabet to represent the blank
        # character. (we suppose the index of blank characters is zero) We use an
        # emoji. There's no need to delete this character; it will automatically remove.
        # Also we suppose the index of blank character is zero
        self.alphabet = "🐱" + self.alphabet
        self.decode_gt = DecodeString(self.map_char_file)
        # Store statistics during the training.
        # The statistics are CER and WER
        self.statistics = {}
        self.whitespace_index = params["dataset"]["whitespace_char_index"]
        self.testing_batch_count = params["training"]["testing_batch_count"]
        self.encode = CodingString(self.map_char_file, used_in_train=False)
        self.decode_string = DecodeString(self.map_char_file)
        self.lr_scheduler = ReduceLROnPlateau(
            self.optimizer,
            "min",
            factor=params["training"]["lr"]["factor"],
            patience=params["training"]["lr"]["patience"],
            verbose=True,
            threshold=params["training"]["lr"]["threshold"],
        )

    def fit(self):
        """
        Train the model.
        """
        torch.autograd.set_detect_anomaly(True)

        for epoch_index in range(self.last_epoch_index, self.max_epoch):
            running_loss = 0.0
            for i, data in enumerate(self.dataloader):
                # Send image and gt batch to the device that is specified.
                imgs = data["img"].to(self.device)
                gts = data["gt"].to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward + backward + optimize
                output = self.model(imgs)
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
                    print(
                        f"Iteration {i} of epoch {epoch_index}) loss: {(running_loss / self.show_log_step):.5f}"
                    )
                    running_loss = 0

            # Test the accuracy of the model at the end of each epoch and step the lr value
            result = self.test(self.testing_batch_count)
            print(
                f"CER value: {result[0]:.5f}, WER value: {result[1]:.5f}, Epoch: {epoch_index}"
            )
            print(f"Ground truth sentence(target): {result[2][1]}")
            print(f"OCRed (predicted) sentence: {result[2][0]}")
            self.statistics[epoch_index] = {"cer": result[0], "wer": result[1]}
            self.lr_scheduler.step(result[0])

            if epoch_index % self.save_check_step == 0:
                out = self.save_checkpoint(epoch_index)
                print(
                    f"Epoch {epoch_index}) Checkpoint saved. checkpoint path: {out}"
                )

    def load_checkpoint(self, file_name: str, lr_val:float=None) -> None:
        """
        Load the checkpoint.
        Checkpoints contain, parameters of the model, optimizer, loss value, and index of the last epoch.

        Parameters
        ----------
        file_name (str): Name of the file. (e.g., file-name.pt)
        """
        checkpoint = torch.load(
            path.join(self.checkpoint_dir, file_name), map_location=self.device
        )
        self.last_epoch_index = checkpoint["lr_scheduler"]["last_epoch"] + 1
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.loss_fn.load_state_dict(checkpoint["loss_state_dict"])
        self.statistics = checkpoint["statistics"]

        # the patience parameter of the scheduler can only change via "params" file
        patience = self.lr_scheduler.state_dict()["patience"]
        checkpoint["lr_scheduler"]["patience"] = patience
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        # Set lr to value user specified
        if lr_val != None:
            self.optimizer.param_groups[0]["lr"] = lr_val

    def save_checkpoint(self, index: int) -> str:
        """
        Parameters
        ----------
        file_name (str): Name of file to store in the dir_path directory.
        (e.g., file-name not file-name-cp-10.pt)(cp: check point)

        Returns
        -------
        Path and name of the file that created
        """
        file_name = self.params["training"]["checkpoint_name"]
        hash_count = file_name.count("#")
        file_name = file_name.replace(
            hash_count * "#", format(index, f"0{hash_count}d")
        )
        file_path = path.join(self.checkpoint_dir, file_name)
        # If file with the same name exist, throw an error
        if Path(file_path).is_file():
            raise FileExistsError(
                f"A file  with the same name and path exist.\nFile name: {file_path}"
            )
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss_state_dict": self.loss_fn.state_dict(),
                "statistics": self.statistics,
                "lr_scheduler": self.lr_scheduler.state_dict(),
            },
            file_path,
        )
        return file_path

    @torch.no_grad()
    def test(self, batch_count=1) -> Tuple[float, float, list]:
        """
        Select ``batch_count`` batches from the test set and calculate WER and CER
        of the batches and return it. Also print a sample of target/predicted pair.

        Parameters
        ----------
        batch_count (int): Number of batches to calculate WER and CER of the model.

        Returns
        -------
        A tuple such that first element is CER value, second element is WER value,
        and the third element is a list as the form [OCRed sentence, target ground truth].
        Note that all sentence are decoded. (i.e., they are string)
        """
        wer = 0
        cer = 0
        sent_numbers = 0
        # First index contains OCRed sentence(sentence predicted by the model),
        # second index is target ground truth. They store as decoded strings.
        sample = list()
        for i in range(batch_count):
            batch = next(iter(self.dataloader))
            output = self.model(batch["img"].to(self.device))
            # Select one sample pair to print to the user
            if i == 0:
                sample.append(
                    viterbi_search(
                        output[0].permute(1, 0).cpu().numpy().astype(np.float32),
                        self.alphabet,
                    )[0]
                )
                # print(sample)
                # print(output.shape)
                sample.append(self.decode_gt(batch["gt"][0]))
                sent_numbers = len(batch) * batch_count

            for out_sent, valid_sent in zip(output, batch["gt"].cpu().numpy()):
                # print(out_sent.shape)
                # Store created sentence by the model and then encode it to integers
                out_sent = viterbi_search(
                        out_sent.permute(1, 0).cpu().numpy().astype(np.float32), 
                        self.alphabet
                )[0]
                out_sent = self.encode(out_sent)
                # print("end out_sent:", out_sent)
                out_sent = clean_sentence(out_sent, 200, 201, 0)
                valid_sent = clean_sentence(valid_sent, 200, 201, 0)
                cer += char_error_rate(out_sent, valid_sent)
                wer += word_error_rate(out_sent, valid_sent, self.whitespace_index)
        cer = cer / sent_numbers
        wer = wer / sent_numbers
        # sample = [self.decode_string(sent) for sent in sample]
        return (cer, wer, sample)

    def get_statistics(self) -> dict:
        """
        Returns all of the stored statistics of the model.
        """
        return self.statistics


class TestModel:
    """
    Test the given model
    """

    def __init__(self, params: dict, model, map_char_file: str):
        """
        Parameters
        ----------
        params (dict): A dictionary contains of the parameters
        model: A object of the model not a class of the model
        device: The device the model is on
        map_char_file (str): Address of the file maps int to char
        """
        self.device = params["training"]["device"]
        self.model = model(params).to(self.device)
        self.decode_string = DecodeString(map_char_file)
        self.params = params
        self.encode = CodingString(map_char_file, used_in_train=False)
        self.alphabet = "".join(load_char_map_file(map_char_file).keys())

    @torch.no_grad()
    def convert_img2text(self, imgs: torch.Tensor) -> tuple:
        """
        Parameters
        ----------
        img (torch.tensor): Image to convert to text. Note that pixel's value
        are in the range of -1 to 1. Note that input should be in the shape (N, C, H, W).
        (N: Size of batch, C: Channels, H: Height, W: Width).

        Returns
        -------
        A tuple with two list. First return contains a list of strings(decoded)
        created by the model (e.g., ["hello", "hi"]). The second return contains
        list of encoded texts produced by the model (e.g., [[1, 20, 3], [66, 5, 2]]).
        """
        # Predicted gts returned from the model
        pred_gts = self.model(imgs.to(self.device))

        # Add a character is not used in the alphabet to represent the blank
        # character. (we suppose the index of blank characters is zero) We use an
        # emoji. There's no need to delete this character; it will automatically remove.
        # Also we suppose the index of blank character is zero
        alphabet = "🐱" + alphabet
        decoded_out_sents = []
        encoded_out_sents = []
        for sent in pred_gts:
            seq, path = viterbi_search(
                sent.permute(1, 0).cpu().numpy().astype(np.float32), alphabet
            )
            decoded_out_sents.append(seq)
            encoded_out_sents.append(self.encode(seq))
        return decoded_out_sents, encoded_out_sents

    def load_checkpoint(self, checkpoint_path: str) -> None:
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

        return F.ctc_loss(
            preds,
            targets,
            pred_lengths,
            target_lengths,
            blank=self.blank,
            zero_infinity=True,
        )
