from genericpath import isdir
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize
import numpy as np
from .create_pairs import CreateImgGtPair
from typing import Union
from glob import glob
from os import path
import csv
from PIL import Image


class OCRDataset(Dataset):
    """
    Manage retrieving, transforming, and returning image/gt pairs.
    Pairs stored in the disk and load only pair we want to the memory.
    """

    def __init__(self, params, used_in:str):
        """
        Parameters
        ----------
        params (dict): The dict contains all of the parameters
        used_in (str): In can be one of the three ``train``, ``test``, and ``validation`` 
        strings and means where the dataset is used.
        """
        image_name_format = params["dataset"]["image_name_format"]
        dataset_dir = params["dataset"]["dataset_dir"]
        self.transforms = params["training"]["transforms"]
        self.used_in = used_in
        # List of all directories in the dataset directory.
        dirs = path.join(dataset_dir, "*")
        dirs = glob(dirs)
        # Store path of all image in all of the directories in the dataset_dir.
        # Keys are the name of images and values are name of directory the image is in.
        imgs_path = dict()
        # The file some information of images (e.g., ground truth of images) stored there.
        # Key is the index(str type) of image and value is the gt
        gts = {}
        for i in dirs:
            if path.isdir(i):
                imgs = path.join(params["dataset"]["dataset_dir"], i, "*")
                imgs = glob(imgs)
                for img in imgs:
                    imgs_path[path.split(img)[1]] = i
                
            elif path.split(i)[1] == "INFO.csv":
                with open(i, "r") as f:
                    csv_f = csv.reader(f, delimiter=",")
                    head = True
                    for row in csv_f:
                        if head == False and row[-1] == self.used_in:
                            gt = row[6]
                            if row[6][-1] == "\n": gt = gt[:-1]
                            gts[row[0]] = gt
                        else:
                            head = False
        self.pairs = []
        for gt_index, gt in zip(gts.keys(), gts.values()):
            img_path = image_name_format
            hash_count = img_path.count("#")
            img_path = img_path.replace(
                hash_count * "#", format(int(gt_index), f"0{hash_count}d")
            )
            img_path = path.join(dataset_dir, imgs_path[img_path], img_path)
            self.pairs.append({"img": img_path, "gt": gt})
        self.pair_count = len(self.pairs)
        
    def __len__(self):
        """
        Return number of data points
        """
        return self.pair_count

    def __getitem__(self, data_id):
        """
        Return the img/gt (image/ground truth) pair with the given id
        as a dictionary with two key: img and gt(ground truth).
        type of image is ``np.ndarray``.
        """
        data = self.pairs[data_id].copy()
        data["img"] = np.asarray(Image.open(data["img"], "r"))
        if self.transforms:
            data = self.transforms(data)

        return data

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
    More acurately, convert the image and gt to tensor. Also swap color axis of
    the image because first saxis of the image should represent hcannels of the
    image(This is a transformer)
    """

    def __call__(self, sample):
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


class AdjustImageChannels:
    """
    Check to all images have three channels. If an input image has one channel,
    return an image with three channel such that the first channel of output equals
    to the input and the two oter channels be zero.
    (This is a transformer)
    """

    def __init__(self, used_in_train=True, swap_img_axis=True):
        """
        Parameters
        ----------
        used_in_transformer (bool): When training it should be true and when
        evaluating and testing this parameter should be false.
        swap_img_axis (bool): Input image to the model should have this shape:
        ``[C x H x W]``. (``C``: Number of channels of the image, ``H``: Height
        of the image, ``W``: Width of the image). If his be true, swap image shape
        from ``(H x W x C)`` to ``(C x H x W)``.
        """
        self.swap_img_axis = swap_img_axis
        self.used_in_train = used_in_train

    def __call__(self, sample: Union[dict, torch.Tensor]) -> dict:
        """
        Parameters
        ----------
        sample: Should be a dict or an image. (Not a batch of images). Note that
        first axis of the image should be channels of image. image should has three dimensions.

        Returns
        -------
        Returned image is in the form ``(C x H x W)``.
        """
        if self.used_in_train:
            # swap color axis because
            # numpy image: H x W x C
            # torch image: C x H x W
            if self.swap_img_axis:
                sample["img"] = sample["img"].permute(2, 0, 1)
            shape = sample["img"].shape
            if shape[0] == 1:
                temp = torch.zeros(3, shape[1], shape[2])
                temp[0] = sample["img"]
                sample["img"] = temp
        else:
            if self.swap_img_axis:
                sample = sample.permute(2, 0, 1)
            shape = sample.shape
            if shape[0] == 1:
                temp = torch.zeros(3, shape[1], shape[2])
                temp[0] = sample
                sample = temp
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
    imgs = torch.stack(
        [resize(data["img"], (longest_height, longest_width)) for data in batch], dim=0
    )
    return {"gt": gts, "img": imgs}
