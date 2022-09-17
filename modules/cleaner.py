from genericpath import isdir
import glob
from os import path
import re
from sys import maxsize
import logging
from .utils import load_char_map_file
from numpy.random import randint


class FileCleaner:
    """
    Remove extra(redundant) things from the text file
    """

    def __init__(self, dir: str):
        """
        Parameters
        ----------
        dir (str): the directory text files stored there or the file we want clean it.
        """
        self.files_path = []
        logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
        if path.isdir(dir):
            self.files_path = glob.glob(path.join(dir, "*"))
        else:
            self.files_path = [dir]

    def clean_files(
        self, hard_clean=False, uniq_chars_map_file="", normalize_len_interval=[]
    ):
        """
        Clean all files in the dicrectory with rules definded and store them.

        Parameters
        ----------
        hard_clean (bool): If true, after running the cleaning rules, remove all
        characters except the characters in the ``uniq_chars_map_file`` file.
        uniq_chars_map_file (str): If ``hard_clean`` equals true, set this
        parameter as the path of ``uniq_chars_map_file``.
        normalized_len_interval (list): If this list contains two element and first
        element is smaller than the second, change length of lines to be in the
        interval we defined.
        """
        for file_path in self.files_path:
            with open(file_path, "r") as f:
                file = f.readlines()

            file = file[30:]  # Remove the first 30 lines of the file
            # Concatenate all element to create a unit string
            file = "".join(file)
            # Remove the lines start with bracket([])(e.g., refrences)
            file = re.sub(r"^\[.*].*$", r"", file, maxsize, re.MULTILINE)
            # Remove remaining brackets([]) and their contents
            file = re.sub(r"\[.*\]", r"", file, maxsize, re.MULTILINE)
            file = re.sub(r"[a-zA-Z]+", r"", file, maxsize)  # Remove English characters
            # Replace English number with Farsi number
            for en_char, fa_char in zip(
                [str(i) for i in range(10)],
                ["۰", "۱", "۲", "۳", "۴", "۵", "۶", "۷", "۸", "۹"],
            ):
                file = re.sub(en_char, fa_char, file, maxsize)
            file = re.sub(
                ",", "،", file, maxsize, re.MULTILINE
            )  # Replace English comma with Arabic comma
            file = re.sub(
                "\?", "؟", file, maxsize, re.MULTILINE
            )  # Replace English question mark to Arabic question mark
            file = re.sub(
                r"[ÙïřŠƘƙǘǙǠʙȘș¾ΘΙΠ٭%×Ϙ'~÷ϙϠЙРљµҘә@•\uf031\u202e\u202b\u2005\u00a0]",
                "",
                file,
                maxsize,
            )  # Remove unnecessary characters
            file = re.sub(
                "[–_]", "-", file, maxsize
            )  # Replace some character with HYPHEN-MINUS(-)
            file = re.sub("·", ".", file, maxsize)  # Replace middle dot with dot
            file = re.sub(
                ";", "؛", file, maxsize
            )  # Replace English semicolon with Arabic semicolon
            file = re.sub(
                "[“”]", '"', file, maxsize
            )  # Replace two characers with quotation mark
            file = re.sub("…", "...", file, maxsize)
            file = re.sub("››", "«", file, maxsize)  # Replace a character
            file = re.sub("‹‹", "»", file, maxsize)  # Replace a character
            file = re.sub("[‹›]", "", file, maxsize)  # Remvoe some characters
            file = re.sub(
                r"\s*ص\s*:\s*[۰-۹]+ *", "", file, maxsize
            )  # Remove number of pages (e.g., ص: ۲۳)
            file = re.sub(
                "ى", "ی", file, maxsize, re.MULTILINE
            )  # Replace character code 0x649 with 0x6cc
            file = re.sub(
                "ي", "ی", file, maxsize, re.MULTILINE
            )  # Replace character code 0x64a with 0x6cc
            file = re.sub(
                "ك", "ک", file, maxsize
            )  # Replace ARABIC LETTER KAF with ARABIC LETTER KEHEH
            file = re.sub(
                "۟", "ْ", file, maxsize
            )  # Replace some characters wit Arabic Sukun
            file = re.sub(
                "\t", " ", file, maxsize
            )  # Replace tab character with space character
            arabic_chars = "ۣۢۨٙ"
            arabic_chars += "۪۠ۧ"
            arabic_chars += "ؙۭ۫۬ۜ"
            file = re.sub(
                f"[{arabic_chars}]", "", file, maxsize
            )  # Remove some Arabic characters
            file = re.sub(
                "\([۰-۹]*\)", "", file, maxsize
            )  # Remove references indeices (e.g., "(23)"")
            if hard_clean:
                uniq_chars = load_char_map_file(uniq_chars_map_file).keys()
                uniq_chars = "".join(uniq_chars)
                # If there was bracket characters in the uniq_chars convert it
                # to an appropriate format to use in Regexp.
                uniq_chars = uniq_chars.replace("[", "\[")
                uniq_chars = uniq_chars.replace("]", "\]")
                # Remove all characters except the characters in the uniq_chars_map_file + \n character
                file = re.sub(f"[^{uniq_chars}\n]+", "", file, maxsize)
            file = re.sub("^\s+", "", file, maxsize, re.MULTILINE)  # Remove empty lines

            # If there exist an end line character at the end of file, remove it
            if file[-1] == "\n":
                file = file[:-1]
            if len(normalize_len_interval) == 2:
                # Replace new line character with space character
                file = re.sub("\n", " ", file, maxsize)
                file_size = len(file) - 1
                new_file = ""
                last_index = 0
                while True:
                    if last_index > file_size:
                        break
                    line_length = randint(
                        normalize_len_interval[0], normalize_len_interval[1] + 1
                    )
                    while (last_index + line_length) < file_size and file[
                        last_index + line_length
                    ] != " ":
                        line_length += 1
                    new_file += file[last_index : (last_index + line_length)] + "\n"
                    last_index += line_length
                file = new_file[
                    :-1
                ]  # Remove additional new line character at the end of the file

            with open(file_path, "w") as f:
                f.write(file)
            logging.info(f"The file cleaned: {path.split(file_path)[1]}")

    def extract_uniq_chars(self) -> set:
        """
        Extract unique chars that are in the files of the directory.

        Returns
        -------
        Extracted uniq chars as a set.
        """
        uniq_chars = set()
        for file_path in self.files_path:
            with open(file_path, "r") as f:
                book = f.read()
            uniq_chars = uniq_chars.union(set(book))
            logging.info(f"Unique chars extracted from: {path.split(file_path)[1]}")

        return uniq_chars
