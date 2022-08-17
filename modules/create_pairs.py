from random import randint
from PIL import Image, ImageFont, ImageDraw, ImageChops, ImageMorph, ImageEnhance
from os import path
from modules.utils import random_from_list
from numpy import array


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

    # def create_pair(self):
    #     """
    #     Create a img/gt pair with random noise, morphology parameter, length,
    #     font size, font, and text.

    #     Returns
    #     -------
    #     Return a list such that the first element represents the image, the second
    #     element represents its gt, and the third element contains details of created
    #     image. Type of the returned iamges is float with values between [0, 255].
    #     """
    #     # Detail of created image store to "detail" var
    #     detail = ""
    #     font_size = 35
    #     fontlist = self.params["fontlist"]
    #     img_name = uuid.uuid4().hex[:12].upper()
    #     # Value of brightness, saturation, and hue to apply on the image
    #     bsh_val = f"{str(self.params['brightness'])}-{str(self.params['saturation'])}-{str(self.params['hue'])}"
    #     img_path = f"{path.join(self.root_directory, 'img', img_name + '.jpg')}"
    #     str_length = randint(1, 22)
    #     gt = ""  # Ground truth store here
    #     for _ in range(str_length):
    #         gt += self.wordlist[randint(0, self.wordlist_length - 1)] + " "
    #     gt = gt[:-1]  # Remove additional space at the end of gt
    #     font_name = fontlist[randint(0, len(fontlist) - 1)]
    #     # command = [
    #     #     "convert",
    #     #     "-background",
    #     #     "white",
    #     #     "-fill",
    #     #     "black",
    #     #     "-font",
    #     #     font_name,
    #     #     "-pointsize",
    #     #     str(font_size),
    #     #     f"pango:{gt}",
    #     #     img_path
    #     #     # "-channel",
    #     #     # "RGB",
    #     #     # "-colorspace",
    #     #     # "RGB",
    #     # ]
    #     command = [
    #         "./create_Data/bin/create_image",
    #         "-font",
    #         font_name,
    #         "-text",
    #         gt,
    #         "-pointsize",
    #         str(font_size),
    #         "-background",
    #         self.params["background_list"][randint(0, len(self.params["background_list"]) - 1)],
    #         "-bsh",
    #         f"{bsh_val}",
    #         "-pos-xy",
    #         f"{x_pos}-{y_pos}",
    #         "-blur",
    #         f"{0-1}",
    #         "-outname",
    #         img_path,
    #         "-size",
    #         img_size,
    #     ]
    #     # Add morphology parameter to the command
    #     # List of all morphology types (morph types and kernels)
    #     morphology_keys = list(self.params["morphology_types"].keys())
    #     # Select as random, one of the morphologies
    #     morph = morphology_keys[randint(0, len(morphology_keys) - 1)]
    #     if morph != "non_morphology":
    #         # Create morph string with appropriate format(e.g., Dilate-Plus-1)
    #         morph = f"{morph}-{self.params['morphology_types'][morph]}"
    #         command.insert(1, "-morphology")
    #         command.insert(2, morph)
    #     subprocess.run(command)

    #     # Read created image from disk and then remove created image from disk
    #     img = Image.open(img_path, "r")
    #     img = np.array(img, dtype=np.float32)
    #     remove(img_path) # Remove created image
    #     return (img, gt, " ".join(command))

    def create_pair(self):
        """
        Create a img/gt pair with random noise, morphology parameter, length,
        font size, font, and text.
        Note: Returned image is converted to a Numpy array.

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
        image = array(image)  # Convert PIL image to Numpy array
        details = f"""fontname: {path.split(font_path)[1]}, fontsize: {font_size}, 
morphology: {morph_type}, brightness: {brightnenss_value}"""
        return (image, gt, details)
