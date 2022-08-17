from random import randint
from bidi.algorithm import get_display
import arabic_reshaper
import matplotlib.pyplot as plt
from torch import Tensor


def random_from_list(input_list):
    """
    Select one element from the input_list as random and return it.
    """
    return input_list[randint(0, len(input_list) - 1)]


def show_imgs(imgs, gts, details, permute=False):
    """
    Show image batches from a tensor. Note that the first dimension of each image is
    width, the second is height, and the third is the number of the image's channel.
    Parameters
    ----------
    imgs(list): A list of images; the type of each image should be float Numpy or
    Torch tensor. Pixel values should be in [0, 1].
    gts(list): A list of ground truths corresponding to images.
    details(list): A list of details about parameters of every image.
    permute(bool): If true, permute each image such that the dimensions correspond
    to the description mentioned at first. (Works on torch.Tensor types only)
    """
    n_imgs = len(imgs)  # Number of input images
    fig, axes = plt.subplots(n_imgs, figsize=(50, 10))
    for i, img in enumerate(imgs, start=0):
        # Convert gt text to a printable style
        gt = arabic_reshaper.reshape(gts[i])
        gt = get_display(gt)
        gt += "\n" + details[i]
        if type(img) == Tensor and permute:
            img = img.permute(2, 0, 1)
        elif type(img) != Tensor and permute:
            raise TypeError(
                f'"permute=True" works just on torch.Tensor type, not on {type(img)}.'
            )

        if n_imgs > 1:
            axes[i].imshow(img)
            axes[i].set_title(gt, loc="right", fontsize=30)
        elif n_imgs == 1:
            axes.imshow(img, cmap="binary")
            axes.set_title(gt, loc="right", fontsize=30)
        else:
            raise ValueError("The imgs list is empty")
