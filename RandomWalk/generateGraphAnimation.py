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
    # filepaths
    fp_in = "mixingPics/*"
    fp_out = "graphMovie.gif"
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