#!/usr/bin/env python3

'''
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Website:  https://maxsitt.github.io/insect-detect-docs/
License:  GNU AGPLv3 (https://choosealicense.com/licenses/agpl-3.0/)

This Python script does the following:
- try to open all .jpg images in the data folder and calculate mean/min/max image width/height
- if an image cannot be opened, move it to the folder "corrupt images/{timestamp}"
'''

import time
from datetime import datetime
from pathlib import Path

from PIL import Image

# Set start time for script execution timer
start_time = time.monotonic()

# Get images (.jpg) from data folder (+ subdirectories)
images = list(Path(".").glob("data/**/*.jpg"))
num_images = len(images)
print(f"\nFound {num_images} .jpg images in data folder.\n")

# Create folder to move corrupt images to
timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S")
Path(f"corrupt_images/{timestamp}").mkdir(parents=True, exist_ok=True)

# Create empty lists to save image widths/heights and paths of corrupt images
img_widths = []
img_heights = []
corrupt_images = []

# Try to open and verify images + save width/height, if not possible move corrupt image
for jpg in images:
    try:
        with Image.open(jpg) as im:
            im.verify()
            img_widths.append(im.width)
            img_heights.append(im.height)
    except Exception:
        print(f"Corrupt image: {jpg}")
        corrupt_images.append(jpg)
        Path(jpg).rename(f"corrupt_images/{timestamp}/{jpg.name}")

# Calculate mean/min/max image width and height
img_width_mean = round(sum(img_widths) / len(img_widths))
img_width_min = min(img_widths)
img_width_max = max(img_widths)
img_height_mean = round(sum(img_heights) / len(img_heights))
img_height_min = min(img_heights)
img_height_max = max(img_heights)

# Print info
num_corrupt_images = len(corrupt_images)
path_corrupt_images = Path.cwd() / f"corrupt_images/{timestamp}"
print(f"\nMoved {num_corrupt_images} corrupt images to {path_corrupt_images}.\n")
print(f"Mean image width:  {img_width_mean}")
print(f"Mean image height: {img_height_mean}\n")
print(f"Min./max. image width:  {img_width_min} / {img_width_max}")
print(f"Min./max. image height: {img_height_min} / {img_height_max}")
print(f"\nScript run time: {round((time.monotonic() - start_time) / 60, 3)} min")
