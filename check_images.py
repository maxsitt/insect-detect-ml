#!/usr/bin/env python3

'''
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Website:  https://maxsitt.github.io/insect-detect-docs/
License:  GNU AGPLv3 (https://choosealicense.com/licenses/agpl-3.0/)

This Python script does the following:
- try to open all .jpg images in the data folder and calculate
  mean/median/min/max and 90th percentile of image width/height
- if an image cannot be opened, move it to the folder "{timestamp}_corrupt_images"
- optional: calculate RGB mean value and standard deviation of all verified images

- optional arguments:
  "-data [path]" path to data folder containing images
  "-rgb" additionally calculate RGB mean value and standard deviation of all images
'''

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, median

import numpy as np
from PIL import Image

# Set start time for script execution timer
start_time = time.monotonic()

# Define optional arguments
parser = argparse.ArgumentParser()
parser.add_argument("-data", "--data_path", type=str,
    help="path to data folder containing images")
parser.add_argument("-rgb", "--rgb_mean_std", action="store_true",
    help="calculate RGB mean value and standard deviation of all images")
args = parser.parse_args()

# Get images (.jpg) from data folder (+ subdirectories)
if args.data_path is not None:
    data_location = Path(args.data_path)
else:
    data_location = Path.cwd()

images = list(data_location.glob("**/*.jpg"))
num_images = len(images)

if num_images > 0:
    print(f"\nFound {num_images} images in {data_location}\n")
else:
    print(f"\nCould not find any images in {data_location}\n")
    sys.exit()

# Create empty lists to save image widths/heights and paths of corrupt images
img_widths = []
img_heights = []
corrupt_images = []

# Try to open all images + save width/height, if not possible save as corrupt image
counter = 0
for image in images:
    try:
        with Image.open(image) as img:
            counter += 1
            print(f"\rOpen image {counter} of {num_images}.", end="")
            img_widths.append(img.width)
            img_heights.append(img.height)
    except Exception:
        corrupt_images.append(image)

# Move corrupt images to folder
if len(corrupt_images) > 0:
    timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S")
    path_corrupt_images = f"{data_location.parent}/{timestamp}_corrupt_images"
    Path(path_corrupt_images).mkdir(parents=True, exist_ok=True)
    for image in corrupt_images:
        Path(image).rename(f"{path_corrupt_images}/{image.name}")
    print(f"\n\nMoved {len(corrupt_images)} corrupt images to {path_corrupt_images}")

# Print info
print(f"\n\nMean image width:    {round(mean(img_widths))} (min: {min(img_widths)} / max: {max(img_widths)})")
print(f"Mean image height:   {round(mean(img_heights))} (min: {min(img_heights)} / max: {max(img_heights)})")
print(f"\nMedian image width:  {round(median(img_widths))}")
print(f"Median image height: {round(median(img_heights))}")
print(f"\n90th percentile of image width:  {round(np.percentile(img_widths, 90))}")
print(f"90th percentile of image height: {round(np.percentile(img_heights, 90))}")

if args.rgb_mean_std:
    # Calculate RGB mean value and standard deviation of all images in the data folder
    # source: https://stackoverflow.com/questions/73350133/how-to-calculate-mean-and-standard-deviation-of-a-set-of-images
    print("\nCalculating RGB mean value and standard deviation of the images...")
    images_checked = list(data_location.glob("**/*.jpg"))
    rgb_values = np.concatenate([Image.open(image).getdata() for image in images_checked], axis=0) / 255
    rgb_mean = tuple(np.around(np.mean(rgb_values, axis=0), 3))
    rgb_std = tuple(np.around(np.std(rgb_values, axis=0), 3))
    print("\nRGB mean value of images:         ", end="")
    print(*rgb_mean, sep=", ")
    print("RGB standard deviation of images: ", end="")
    print(*rgb_std, sep=", ")

# Print script run time
print(f"\nScript run time: {round((time.monotonic() - start_time) / 60, 3)} min\n")
