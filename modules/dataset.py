import torch
from torch.utils.data import Dataset
import glob
from os.path import join, basename, splitext
from os import sep
from skimage import io
import matplotlib.pyplot as plt
from torchvision.transforms.functional import resize
import numpy as np

class OCRDataset(Dataset):
    def __init__(self, params):
        """
        Find all images from img/ directory.
        root_dir: Root path of the data
        text_int_map_path: Path of the file contains map between chars and ints
        """
        root_dir = params["training_params"]["dataset_root_path"]
        self.transforms = params["training_params"]["img_transforms"]
        self.gt_dir = f"""{join(root_dir, "gt")}"""
        self.img_dir = f"""{join(root_dir, "img")}"""
        self.all_img_list = glob.glob(f"{self.img_dir}{sep}*")

    def __len__(self):
        """
        Return number of data points
        """
        return len(self.all_img_list)

    def __getitem__(self, data_id):
        """
        Return the img/gt (image/ground truth) pair with the given id
        as a dictionary with two key: img and gt(ground truth).
        """
        if torch.is_tensor(data_id):
            data_id = data_id.tolist()
        image = io.imread(join(self.img_dir, basename(self.all_img_list[data_id])))
        image = np.float32(image)
        # Extract name of file without extension
        gt_path = splitext(basename(self.all_img_list[data_id]))[0]
        gt_path = join(self.gt_dir, f"{gt_path}.txt")
        with open(gt_path, "r") as f:
            # Convert the string to a numpy array of one-hat vectors
            # Ignore the new line character("\n" or 0xa) at the end of line
            gt = f.readlines()[0][:-1]

        pair = {"img": image, "gt": gt}
        if self.transforms:
            pair = self.transforms(pair)

        return pair

class CodingString:
    """
    Map the characters of a string to corresponding ints and return it.
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

    def _one_hot_vector(self, index):
        """
        Create one-hot vector for the given character and return.
        index: Index of the one-hot vector to have value one.

        Returns
        -------
        one-hot vector(Numpy array)
        """
        one_hot = np.zeros(self.vocab_size, dtype=int)
        one_hot[index] = 1

        return one_hot

class Normalize:
    """
    Rescale value of pixels to have value between 0 and 1 and then rescale again 
    to pixels have value between -1 and +1.
    (This is a transformer)
    """
    def __call__(self, sample):
        sample["img"] = ((sample["img"] / 255) - 0.5) / 0.5
        return sample

class ToTensor:
    """
    Convert given samples to Tensors.
    More acurately, convert the image and gt to tensor.
    (This is a transformer)
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

def show_img(tensor):
    """
    Show image batches from a tensor
    """
    if tensor.dim() == 4:
        for img in tensor:
            # Move the channel to be the last dimension
            plt.imshow(img.permute(1, 2, 0))
            plt.show()
    elif tensor.dim() == 3:
        plt.imshow(img.permute(1, 2, 0))


def create_char_to_int_map_file(unique_char_file, map_file):
    """
    Create a map file(i.e., a file contains unique chars and their corresponding
    integer) from file contain uniqie characters.
    """
    with open(unique_char_file, "r") as f:
        uniq_file = f.readlines()
    with open(map_file, "w") as f:
        for index, line in enumerate(uniq_file):
            # Split the unique characters and assign to them an unique numberand save them
            f.write(f"{line[0]}#{format(index, '03d')}\n")


def dataloader_collate_fn(batch):
    """
    Merge a list of samples(batch) such that every ground truth in samples
    have the same dimension.
    """
    # Merge ground truth such that they have the same dimension
    longest_gt = max(data["gt"].shape[0] for data in batch)
    gts = torch.zeros((len(batch), longest_gt))
    for i, data in enumerate(batch):
        gts[i][:len(data["gt"])] = data["gt"]

    imgs = torch.stack([data["img"] for data in batch], dim=0)
    return {"gt": gts, "img": imgs}
