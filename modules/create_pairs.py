import subprocess
from threading import Thread, Lock
from random import randint, random
import uuid
from os import path, remove
from .params import params
from PIL import Image
import numpy as np

class CreateImgGtPair:
    def __init__(self, root_directory, wordlist_file):
        """
        Parameters
        -----------
        root_directory (string): The path of directory that img directory is inside that.
        wordlist_file (string): The name of wordlist that is inside the root 
        directory and the words selected from this file. Note that in the wordlist, 
        each word should be placed in a separate line.
        """
        with open(path.join(root_directory, wordlist_file), "r") as f:
            self.wordlist = f.readlines()
            # Remove newline character at the end of each word
            self.wordlist = [word.rstrip() for word in self.wordlist]
        self.wordlist_length = len(self.wordlist)
        # List of morphology types
        self.morphology_types = {
            "Open Disk:": randint(1, 6),
            "Close Octagon:": randint(1, 3),
            "Erode Plus:": randint(1, 6),
            "Dilate Plus:": randint(1, 4),
            "Dilate Disk": randint(1, 4),
            "non_morphology": "",
        }
        self.root_directory = root_directory
        
    def create_pair(self, index):
        """
        Create a img/gt pair with random noise, morphology parameter, length, font size, font, and text.
        """
        font_size = randint(20, 40)
        fontlist = params["articifial_dataset"]["fontlist"]
        img_name = uuid.uuid4().hex[:12].upper()
        img_path = f"{path.join(self.root_directory, 'img', img_name + '.jpg')}"
        str_length = randint(1, 22)
        gt = "" # Ground truth store here
        for _ in range(str_length):
            gt += self.wordlist[randint(0, self.wordlist_length-1)] + " "
        gt = gt[:-1] # Remove additional space at the end of gt
        command = [
            "convert",
            "-background",
            "red",
            "-fill",
            "black",
            "-channel",
            "RGB",
            "-colorspace",
            "RGB",
            "-font",
            fontlist[randint(0, len(fontlist)-1)],
            "-pointsize",
            font_size,
            f"pango:{gt}",
            img_path,
        ]
        subprocess.run(command)
        
        # Read created image from disk and then remove created image from disk
        img = Image.open(img_path, "r")
        img = np.array(img, dtype=np.float32)
        remove(img_path) # Remove created image
        return
        
        