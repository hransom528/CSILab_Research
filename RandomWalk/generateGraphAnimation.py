# Harris Ransom
# Graph Animation Generator
# 10/15/2024

# Imports
import argparse
import imageio.v2 as imageio
import glob
import re

# MAIN
if __name__ == "__main__":
    images = []
    filenames = glob.glob('mixingPics/*')
    filenames.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    print(filenames)
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave('graphMovie.gif', images)