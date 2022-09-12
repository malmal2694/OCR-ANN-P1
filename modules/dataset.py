import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize
import numpy as np
from .create_pairs import CreateImgGtPair


class OCRDataset(Dataset):
    """
    Manage creating, transforming, and returning image
    """

    def __init__(self, params):
        """
        params (dict): The dict contains all of the parameters
        """
        self.pair = CreateImgGtPair(params["artificial_dataset"])
        self.transforms = params["training_params"]["transforms"]
        self.image_numbers = params["artificial_dataset"]["image_numbers"]

    def __len__(self):
        """
        Return number of data points
        """
        return self.image_numbers

    def __getitem__(self, data_id):
        """
        Return the img/gt (image/ground truth) pair with the given id
        as a dictionary with two key: img and gt(ground truth).
        """
        image, gt, _ = self.pair.create_pair()
        pair = {"img": image, "gt": gt}
        if self.transforms:
            pair = self.transforms(pair)

        return pair


class CodingString:
    """
    Map the characters of a string to corresponding ints and return it as a list.
    (This is a transformer)
    """

    def __init__(self, map_char_file):
        """
        map_char_file: The path of the file that map chars to ints
        """
        with open(map_char_file, "r") as f:
            uniq_file = f.readlines()
        # Create a dict that maps unique chars to ints
        self.char_to_int_map = {}
        for line in uniq_file:
            self.char_to_int_map[line[0]] = int(line[2:5])
        # vocab_size is the number of unique chars plus one that represent blank char.
        # Refer to CTC loss algorithm.
        self.vocab_size = len(self.char_to_int_map)  # + 1

    def __call__(self, sample):
        """
        Map the string to a numpy array of ints and then retrurn this array.

        Parameters
        ----------
        sample(dict): the sample we want to map its ground truth(as a tensor object)
        to int and then return the whole sample.
        """
        txt_out = np.array([], dtype=int)
        for char in sample["gt"]:
            txt_out = np.append(txt_out, self.char_to_int_map[char])
        sample["gt"] = txt_out
        return sample

class Normalize:
    """
    Rescale value of pixels to have value between 0 and 1 and then rescale again
    to pixels have value between -1 and +1.
    (This is a transformer)
    """
    def __init__(self, used_in_train=True):
        """
        used_in_transformer (bool): when training it should be true and when 
        evaluating and testing this parameter should be false.
        """
        self.used_in_train = used_in_train
        
    def __call__(self, sample):
        if self.used_in_train == True:
            sample["img"] = ((sample["img"] / 255) - 0.5) / 0.5
        else:
            sample = ((sample / 255) - 0.5) / 0.5
        return sample


class ToTensor:
    """
    Convert given samples to Tensors.
    More acurately, convert the image and gt to tensor. (This is a transformer)
    """

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        sample["img"] = sample["img"].transpose((2, 0, 1))
        return {
            "img": torch.from_numpy(sample["img"]),
            "gt": torch.from_numpy(sample["gt"]),
        }


class Resize:
    """
    A class for resizing images
    (This is a transformer)
    """

    def __init__(self, size):
        """
        Parameters
        ----------
        size (tuple or list): Size of returned image
        """
        self.size = size

    def __call__(self, sample):
        sample["img"] = resize(sample["img"], self.size)
        return sample

def dataloader_collate_fn(batch):
    """
    Merge a list of samples(batch) such that every ground truth in the samples
    have the same dimension.
    gts in each batch have the same length.
    """
    # Merge ground truth such that they have the same dimension
    longest_gt = max(data["gt"].shape[0] for data in batch)
    gts = torch.zeros((len(batch), longest_gt))
    for i, data in enumerate(batch):
        gts[i][: len(data["gt"])] = data["gt"]

    longest_height = max(data["img"].shape[1] for data in batch)
    longest_width = max(data["img"].shape[2] for data in batch)
    imgs = torch.stack([resize(data["img"], (longest_height, longest_width)) for data in batch], dim=0)
    return {"gt": gts, "img": imgs}
