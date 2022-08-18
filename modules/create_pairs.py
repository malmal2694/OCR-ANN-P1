from random import randint
from PIL import Image, ImageFont, ImageDraw, ImageChops, ImageMorph, ImageEnhance
from os import path
from modules.utils import random_from_list
from numpy import array, float32


class CreateImgGtPair:
    def __init__(self, params):
        """
        Parameters
        -----------
        params (dict): Artificial dataset parameters (e.g., params["artificial_dataset"])
        """
        with open(params["wordlist_path"], "r") as f:
            self.wordlist = f.readlines()
            # Remove newline character at the end of each word
            self.wordlist = [word.rstrip() for word in self.wordlist]
        self.params = params

    def create_pair(self):
        """
        Create a img/gt pair with random noise, morphology parameter, length,
        font size, font, and text.
        Note: Returned image is converted to a Numpy.flaot32 array.

        Returns
        -------
        Return a list such that the first element represents the image, the second
        element represents its gt, and the third element contains details of the created
        image. Type of the returned iamges is float with values between [0, 255].
        """
        font_size = 35
        str_length = randint(1, 22)
        gt = ""  # Ground truth store here
        for _ in range(str_length):
            gt += random_from_list(self.wordlist) + " "
        gt = gt[:-1]  # Remove additional new-line character at the end of gt
        font_path = random_from_list(self.params["fontlist"])
        print(f"(create_pairs.py) font path: {font_path}")
        font = ImageFont.truetype(font_path, font_size)
        # Calculate size of the text (Width and height)
        bbox = font.getbbox(gt, direction="rtl")
        width_txt = bbox[2] - bbox[0]
        height_txt = bbox[3] - bbox[1]
        background_img = Image.open(random_from_list(self.params["background_list"]))

        # Create image of text with a background image
        image = Image.new(
            "L", (width_txt, height_txt + int(0.33 * height_txt)), color=(255)
        )
        draw = ImageDraw.Draw(image)
        draw.text((0, 1 / 6 * height_txt), gt, font=font, fill="black", direction="rtl")

        # Apply morphology on the image
        morph_type = random_from_list(self.params["morphology_types"])
        if morph_type != []:
            lb = ImageMorph.LutBuilder(morph_type)
            lut = lb.build_lut()
            morph_op = ImageMorph.MorphOp(lut)  # Morphology operation
            _, image = morph_op.apply(image)

        image = ImageChops.multiply(image.convert("RGB"), background_img)
        brightness_image = ImageEnhance.Brightness(image)
        brightnenss_value = self.params["brightness"]
        image = brightness_image.enhance(brightnenss_value)
        image = array(image, dtype=float32)  # Convert PIL image to the Numpy.flaot32 array
        details = f"""fontname: {path.split(font_path)[1]}, fontsize: {font_size}, 
morphology: {morph_type}, brightness: {brightnenss_value}"""
        return (image, gt, details)
