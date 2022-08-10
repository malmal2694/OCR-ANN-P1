import torch
from torch.utils.data import Dataset
import glob
from os.path import join, basename, splitext
from os import sep
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from modules import params


class OCRDataset(Dataset):
    def __init__(self, params):
        """
        Find all images from img/ directory.
        root_dir: Root path of the data
        text_int_map_path: Path of the file contains map between chars and ints
        """
        root_dir = params["training_params"]["dataset_root_path"]
        text_int_map_file = params["training_params"]["uniq_chars_map"]
        self.transforms = params["training_params"]["img_transforms"]

        self.gt_dir = f"""{join(root_dir, "gt")}"""
        self.img_dir = f"""{join(root_dir, "img")}"""
        self.all_img_list = glob.glob(f"{self.img_dir}{sep}*")
        self.text_int_map = TextToTensor(text_int_map_file)

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
        # Extract name of file without extension
        gt_path = splitext(basename(self.all_img_list[data_id]))[0]
        gt_path = join(self.gt_dir, f"{gt_path}.txt")
        with open(gt_path, "r") as f:
            # Convert the string to a numpy array of one-hat vectors
            # Ignore the new line character("\n") at the end of line
            gt = self.text_int_map.convert(f.readlines()[0][:-1])

        pair = self.to_tensor({"img": image, "gt": gt})
        if self.transforms:
            pair["img"] = self.transforms(pair["img"])

        return pair

    def to_tensor(self, sample):
        """
        Convert numpy arrays in the sample to Tensors.
        More acurately, convert the image and gt to tensor.
        """
        image, gt = sample["img"], sample["gt"]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {
            "img": torch.from_numpy(image).to(torch.float32),
            "gt": torch.from_numpy(gt),
        }


class TextToTensor:
    """
    Map the characters of a string to corresponding ints and then create a
    one-hot-vector for this array and then return it.
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

    def convert(self, text):
        """
        Map the string to a numpy array of ints and then return one-hot vectors of this array.
        text: text we want to convert
        """
        out = np.array([], dtype=int)
        for char in text:
            out = np.append(
                out, self._one_hot_vector(self.char_to_int_map[char]), axis=0
            )
        # Resize "out" to be 2D; In other words, each row represent one-hat vector for a character
        return np.reshape(out, (len(text), self.vocab_size))

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
    longest_gt = max(data["gt"].shape[0] for data in batch)
    # Merge ground truth such that they have the same dimension
    gts = torch.zeros(
        (len(batch), longest_gt, params.params["training_params"]["vocab_size"])
    )
    for i, data in enumerate(batch):
        for j, char_one_hot_vector in enumerate(data["gt"], start=0):
            gts[i][j] = char_one_hot_vector

    imgs = torch.stack([data["img"] for data in batch], dim=0)
    return {"gt": gts, "img": imgs}
