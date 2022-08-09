from operator import getitem
import torch
from torch.utils.data import Dataset
import glob
from os.path import join, basename, splitext
from os import sep
from skimage import io
import matplotlib.pyplot as plt


class OCRDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        """
        Find all images from img/ directory.
        root_dir: Path of data
        """
        self.transforms = transforms
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
        Return the img/gt (image/ground truth) pair with given id
        as a dictionary with two key: img and gt(ground truth).
        """
        if torch.is_tensor(data_id):
            data_id = data_id.tolist()

        image = io.imread(join(self.img_dir, basename(self.all_img_list[data_id])))
        # Extract name of file without extension
        gt_path = splitext(basename(self.all_img_list[data_id]))[0]
        gt_path = join(self.gt_dir, f"{gt_path}.txt")
        with open(gt_path, "r") as f:
            gt = f.readlines()[0]

        if self.transforms:
            pair = self.to_tensor({"img": image, "gt": gt})
            pair["img"] = self.transforms(pair["img"])
            return pair
        return self.to_tensor({"img": image, "gt": gt})

    def to_tensor(self, sample):
        """
        Convert numpy arrays in the sample to Tensors.
        More acurately, convert the image to tensor and remain "gt" without change.
        """
        image, gt = sample['img'], sample['gt']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        # print(image.shape)
        image = image.transpose((2, 0, 1))
        return {'img': torch.from_numpy(image).to(torch.float32), 'gt': gt}
    
def show_img(images):
    """
    Show image batches from a tensor
    """
    if images.dim() == 4:
        for img in images:
            # Move the channel to be the last dimension
            plt.imshow(img.permute(1, 2, 0))
            plt.show()
    elif images.dim() == 3:
        plt.imshow(img.permute(1, 2, 0))