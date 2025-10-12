# stdlib imports
import os
import sys

# pip imports
import numpy as np
import matplotlib.pyplot

# local imports
import mpl_graph

__dirname__ = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(__dirname__, "../output")


class ExamplesUtils:
    @staticmethod
    def preamble():
        """
        If in testing mode, set random seeds for reproducibility.
        """
        # if not in testing mode, return now
        MPLSC_TESTING = os.environ.get("MPLSC_TESTING", "False")
        if MPLSC_TESTING != "True":
            return

        # set random seed for reproducibility
        np.random.seed(0)
        mpl_graph.core.random.Random.set_random_seed(0)

    @staticmethod
    def postamble():
        """
        If in testing mode, save the current matplotlib.pyplot figure to the output directory
        and return True to indicate that the calling script should exit.
        Otherwise, return False to indicate that the calling script should continue.
        """
        should_exit = False

        # if not in testing mode, return now
        MPLSC_TESTING = os.environ.get("MPLSC_TESTING", "False")
        if MPLSC_TESTING != "True":
            return should_exit

        # mark that we should exit after saving the image
        should_exit = True

        # get the __file__ of the calling script
        example_filename = getattr(sys.modules.get("__main__"), "__file__", None)
        assert example_filename is not None, "Could not determine example filename"

        # Extract example basename and directory
        example_basename = os.path.basename(example_filename).replace(".py", "")

        # save the output to a file
        image_path = os.path.join(output_path, f"{example_basename}.png")
        matplotlib.pyplot.savefig(image_path)
        print(f"Saved output to {image_path}")

        return should_exit
