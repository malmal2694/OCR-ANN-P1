from torchvision.transforms import Compose
from .dataset import ToTensor, Normalize, AdjustImageChannels
from .utils import CodingString
from random import randint
import glob
from os import path
from torch import device
from torch.cuda import is_available


unique_chars_map_file = path.join(
    path.abspath(path.dirname(__file__)), "../create-data/unique_chars_map.txt"
)

params = {
    "dataset": {
        "unique_chars_map_file": path.join(
            path.abspath(path.dirname(__file__)), "../create-data/unique_chars_map.txt"
        ),
        # Index of the whitespace(The character that is between words of sentence).
        "whitespace_char_index": 1,
    },
    "artificial_dataset": {
        "fontlist": glob.glob(
            path.join(path.dirname(__file__), ("../create-data/fonts/*"))
        ),  # List fonts
        "morphology_types": [
            # ["4:(111 ..1 111)->1"],
            # ["4:(.00 ..0 .1.)->0"],
            [],
        ],
        # number of word in each line is in the interval
        "gt_length_interval": [4, 22],
        # List of images for background
        "background_list": [""],
        # "background_list": glob.glob(
        #     path.join(path.dirname(__file__), "../create-data/backgrounds/*")
        # ),
        # Iinterval of Brightness of the image (It will divided to 100)
        "brightness": (70, 130),
        # The number of images. Because we don't have a real dataset of image/gt
        # pairs also image/gt pairs create online, the value of this parameter
        # is desirable, but it's better for this value to be large.
        "image_numbers": 4000,
        # The name of wordlist that is inside the root
        # directory and the words selected from this file. Note that in the wordlist,
        # each word should be placed in a separate line.
        "wordlist_path": path.join(
            path.abspath(path.dirname(__file__)), "../create-data/complete.wordlist"
        ),
    },
    "model_params": {
        "transfer_learning": None,  # dict : {model_name: [state_dict_name, checkpoint_path, learnable, strict], }
        "input_channels": 3,  # 1 for grayscale images, 3 for RGB ones (or grayscale as RGB)
        "dropout": 0.5,  # dropout probability for standard dropout (half dropout probability is taken for spatial dropout)
    },
    # List of transforms that apply on the image/gt pair
    "training": {
        "transforms": Compose(
            [
                CodingString(unique_chars_map_file),
                ToTensor(),
                Normalize(),
                AdjustImageChannels(),
            ]
        ),
        # Note that acount space character too. There's no need to add blank character to these chars.
        "vocab_size": 91,
        # Parameters of optimizer and lr scheduler
        "lr": {
            # The learning rate we start with this value
            "start_lr": 0.005,
            # Factor by which the learning rate will be reduced. new_lr = lr * factor.
            "factor": 0.1,
            # Number of epochs with no improvement after which learning rate will be reduced.
            # For example, if patience = 2, then we will ignore the first 2 epochs with no
            # improvement, and will only decrease the LR after the 3rd epoch if the loss
            # still hasn’t improved then.
            "patience": 10,
            # Threshold for measuring the new optimum, to only focus on significant changes.
            "threshold": 5e-3,
        },
        "epoch_numbers": 2000,  # Maximum number of epoches
        "checkpoint_dir": path.join(  # Directory the checkpoint files store and read from
            path.abspath(path.dirname(__file__)), "../create-data/checkpoints"
        ),
        # Name of the checkpoint file. Instead of the # character, the index of the
        # current epoch will replace. The number of the # character shows the
        # length of the number that will replace.
        "checkpoint_name": "checkpoint-####.pt",
        # "max_nb_epochs": 5000,  # max number of epochs for the training
        # "max_training_time": 3600 * (24 + 23),  # max training time limit (in seconds)
        # "load_epoch": "best",  # ["best", "last"], to load weights from best epoch or last trained epoch
        # "interval_save_weights": None,  # None: keep best and last only
        "use_ddp": False,  # Use DistributedDataParallel
        "use_apex": True,  # Enable mix-precision with apex package
        # Use GPU if available else CPU
        "device": device("cuda:0" if is_available() else "cpu"),
        "batch_size": 16,  # mini-batch size per GPU
        # Number of batches to use in testing the model
        "testing_batch_count": 10,
        # "optimizer": {
        #     # "class": Adam,
        #     "args": {
        #         "lr": 0.0001,
        #         "amsgrad": False,
        #     }
        # },
        "eval_on_valid": True,  # Whether to eval and logs metrics on validation set during training or not
        "eval_on_valid_interval": 2,  # Interval (in epochs) to evaluate during training
        "focus_metric": "cer",  # Metrics to focus on to determine best epoch
        "expected_metric_value": "low",  # ["high", "low"] What is best for the focus metric value
        # "set_name_focus_metric": "{}-valid".format(dataset_name),
        # "train_metrics": ["loss_ctc", "cer", "wer"],  # Metrics name for training
        # "eval_metrics": [
        #     "loss_ctc",
        #     "cer",
        #     "wer",
        # ],  # Metrics name for evaluation on validation set during training
        # "force_cpu": False,  # True for debug purposes to run on cpu only
    },
}
