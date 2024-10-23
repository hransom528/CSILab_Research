# Harris Ransom
# Graph Animation Generator
# 10/15/2024

# Imports
import argparse
import contextlib
from PIL import Image
import glob
import re

# MAIN
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
                    prog='GenerateGraphAnimation',
                    description='Generates gif of graph data mixing from collection of images')
    parser.add_argument("-i", "--input", default="mixingPics/*", help="Input image directory path (e.g. 'mixingPics/*')")
    parser.add_argument("-o", "--output", default="graphMovie.gif", help="Output gif file path (e.g. 'graphMovie.gif')")
    args = parser.parse_args()

    # filepaths
    fp_in = args.input
    fp_out = args.output
    glob_in = glob.glob(fp_in)
    glob_in.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

    # use exit stack to automatically close opened images
    with contextlib.ExitStack() as stack:

        # lazily load images
        imgs = (stack.enter_context(Image.open(f))
                for f in glob_in)

        # extract  first image from iterator
        img = next(imgs)

        # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
        img.save(fp=fp_out, format='GIF', append_images=imgs,
                save_all=True, duration=550, loop=0)