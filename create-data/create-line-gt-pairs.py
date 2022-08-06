from distutils import text_file
import subprocess
from threading import Thread, Lock


class WriteLineAsPll:
    """
    A class for handling create line images and their ground truth as parallel(i.e., some thread)
    """

    def __init__(self, n_pll_process, in_file, out_file_name):
        """
        n_pll_process: Number of parallel processes for creating image/gt pairs
        in_file: The file path that line contents store there. each of that file represent a image/gt pairs
        out_file_name: The fixed name assign to every image/gt pairs
        """
        self.in_file = in_file
        self.out_file_name = out_file_name
        self.lock = Lock()
        self.pll_process = [Thread()] * n_pll_process
        self.lines = open(self.in_file, "r").readlines()
        self.l_index = 0  # Line index

    def is_done(self):
        if (self.l_index + 1) == len(self.lines):
            return True
        return False

    def run_command(self):
        """
        Run command to create (just) an image/gt pair
        """
        # Lock the objects
        self.lock.acquire(True)
        txt_file_out = f"{self.out_file_name}{format(self.l_index, '06d')}.txt"
        line_content = self.lines[self.l_index]
        command = [
            "pango-view",
            "--font",
            "Mehr Nastaliq Web 50",
            "-q",
            "-o",
            f"data/img/{self.out_file_name}{format(self.l_index, '06d')}.jpg",
            "--rtl",
            "--language",
            "fa_IR",
            "--margin",
            "23 5 0",
            f"data/{txt_file_out}"
        ]
        self.l_index += 1
        self.lock.release()

        txt_file = open(f"data/gt/{txt_file_out}", "w")
        txt_file.write(line_content)
        txt_file.close()
        subprocess.run(command)

    def spawn_process(self):
        """
        Create a new process to create a new image/gt pair when active threads
        are less than "len(self.pll_process).
        """
        for index, prcs in enumerate(self.pll_process):
            if not prcs.is_alive():
                self.pll_process[index] = Thread(target=self.run_command)
                self.pll_process[index].start()


if __name__ == "__main__":
    write_lines = WriteLineAsPll(8, "data/lines.txt", "mehrnastaliq")
    while not write_lines.is_done():
        write_lines.spawn_process()
