import subprocess
from threading import Thread, Lock


class WriteLineAsPll:
    """
    A class for handling create line images and their ground truth as parallel(i.e., some thread)
    """

    def __init__(self, n_pll_process, in_file, out_file_name, step_index):
        """
        n_pll_process: Number of parallel processes for creating image/gt pairs
        in_file: The file path that line contents store there. each of that file represent a image/gt pairs
        out_file_name: The fixed name assign to every image/gt pairs
        step_ibdex: Show log at every "step_index" step
        """
        self.step_index = step_index
        self.in_file = in_file
        self.out_file_name = out_file_name
        self.pll_process = [Thread()] * n_pll_process
        self.lines = open(self.in_file, "r").readlines()
        self.l_index = 0  # Index of last created line

    def _is_done(self):
        """
        Check if all pair/gt are created?
        """
        if (self.l_index + 1) >= len(self.lines):
            return True
        return False

    def run_command(self):
        """
        Run command to create (just) an image/gt pair
        """
        lock = Lock()
        # Lock the below objects
        lock.acquire(True)
        txt_file_out = f"{self.out_file_name}{format(self.l_index, '06d')}.txt"
        line_content = self.lines[self.l_index]
        command = [
            "convert",
            "-background",
            "white",
            "-fill",
            "black",
            "-font",
            "Mehr Nastaliq Web",
            "-pointsize",
            "50",
            f"pango:{line_content}",
            f"data/img/{self.out_file_name}{format(self.l_index, '06d')}.jpg"
        ]

        # Print index of last created line at every "self.step_index" step
        if (self.l_index % self.step_index) == 0:
            print(f"Creating Line {format(self.l_index, '06d')}")
        txt_file = open(f"data/gt/{txt_file_out}", "w")
        txt_file.write(line_content)
        txt_file.close()
        subprocess.Popen(command)
        lock.release() # Release the key

    def spawn_process(self):
        """
        Create a new process to create a new image/gt pair when active threads
        are less than "len(self.pll_process).
        """
        while not self._is_done():
            for index, prcs in enumerate(self.pll_process):
                if not prcs.is_alive() and not self._is_done():
                    self.l_index += 1
                    self.pll_process[index] = Thread(target=self.run_command)
                    self.pll_process[index].start()

if __name__ == "__main__":
    write_lines = WriteLineAsPll(8, "data/lines.txt", "mehrnastaliq", 10)
    write_lines.spawn_process()
