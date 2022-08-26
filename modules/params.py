from torchvision.transforms import Compose
from .dataset import CodingString, ToTensor, Normalize
from random import randint
import glob
from os import path
from torch import device

params = {
    "unique_chars_map_file": path.join(
        path.abspath(path.dirname(__file__)), 
        "../create-data/unique_chars_map.txt"
    ),
    "artificial_dataset": {
        "fontlist": glob.glob(
            path.join(path.dirname(__file__), 
            ("../create-data/fonts/*"))
        ),  # List fonts
        "morphology_types": [
            # ["4:(111 ..1 111)->1"],
            ["4:(.00 ..0 .1.)->0"],
            # ["0:(111 ..0 0..)->1"],
            # [
            #     "4:(1.0 111 000)->0",
            #     "4:(.0. .1. ...)->1",
            #     "4:(01. .1. ...)->1",
            # ],
            [], [], [], [],
            [],
        ],
        "gt_length_interval": [1, 2],
        # List of images for background
        "background_list": glob.glob(
            path.join(path.dirname(__file__), "../create-data/backgrounds/*")
        ),
        "brightness": randint(70, 130) / 100,  # Brightness of the image
        # The number of images. Because we don't have a real dataset of image/gt
        # pairs also image/gt pairs create online, the value of this parameter
        # is desirable, but it's better for this value to be large.
        "image_numbers": 3000,
        # The name of wordlist that is inside the root
        # directory and the words selected from this file. Note that in the wordlist,
        # each word should be placed in a separate line.
        "wordlist_path": path.join(
            path.abspath(path.dirname(__file__)), 
            "../create-data/complete.wordlist"
        ),
    },
    "model_params": {
        # Model classes to use for each module
        # "models": {
        #     "encoder": FCN_Encoder,
        #     "decoder": Decoder,
        # },
        "transfer_learning": None,  # dict : {model_name: [state_dict_name, checkpoint_path, learnable, strict], }
        "input_channels": 3,  # 1 for grayscale images, 3 for RGB ones (or grayscale as RGB)
        "dropout": 0.5,  # dropout probability for standard dropout (half dropout probability is taken for spatial dropout)
    },
    # List of transforms that apply on the image
    "training_params": {
        "img_transforms": Compose(
            [
                CodingString(),
                ToTensor(),
                # Resize((200, 2000)),
                Normalize(),
            ]
        ),
        "vocab_size": 109,  # Note that acount space character too.
        "min_opt_lr": 0.0004,  # Minimum learning rate we can reach
        "max_opt_lr": 0.005, # Maximum learning rate we can reach
        "epoch_numbers": 2000, # Maximum number of epoches
        "checkpoint_dir": path.join( # Directory the checkpoint files store and read from
            path.abspath(path.dirname(__file__)),
            "../create-data/checkpoints"
        ),
        # Name of the checkpoint file. Instead of the # character, the index of the 
        # current epoch will replace. The number of the # character shows the 
        # length of the number that will replace.
        "checkpoint_name": "checkpoint-####.pt",
        "max_nb_epochs": 5000,  # max number of epochs for the training
        "max_training_time": 3600 * (24 + 23),  # max training time limit (in seconds)
        "load_epoch": "best",  # ["best", "last"], to load weights from best epoch or last trained epoch
        "interval_save_weights": None,  # None: keep best and last only
        "use_ddp": False,  # Use DistributedDataParallel
        "use_apex": True,  # Enable mix-precision with apex package
        # "device": device("cuda:0"), # The device that all operations do on it
        "device": device("cuda:0"),
        # "nb_gpu": torch.cuda.device_count(),
        "batch_size": 16,  # mini-batch size per GPU
        "optimizer": {
            # "class": Adam,
            "args": {
                "lr": 0.0001,
                "amsgrad": False,
            }
        },
        "eval_on_valid": True,  # Whether to eval and logs metrics on validation set during training or not
        "eval_on_valid_interval": 2,  # Interval (in epochs) to evaluate during training
        "focus_metric": "cer",  # Metrics to focus on to determine best epoch
        "expected_metric_value": "low",  # ["high", "low"] What is best for the focus metric value
        # "set_name_focus_metric": "{}-valid".format(dataset_name),
        "train_metrics": ["loss_ctc", "cer", "wer"],  # Metrics name for training
        "eval_metrics": [
            "loss_ctc",
            "cer",
            "wer",
        ],  # Metrics name for evaluation on validation set during training
        "force_cpu": False,  # True for debug purposes to run on cpu only
    },
}
