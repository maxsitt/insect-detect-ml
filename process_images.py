"""Functions to process images and extract image information.

Source:   https://github.com/maxsitt/insect-detect-ml
License:  GNU AGPLv3 (https://choosealicense.com/licenses/agpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

Functions:
    check_images(): Check directory for corrupt .jpg images and save image dimensions to .csv.
    calc_img_stats(): Calculate statistics of image dimensions and save result to .csv.
    calc_rgb_stats(): Calculate RGB mean values and standard deviations and save result to .csv.

    main(): Get .jpg files in directory and check for corrupt images, write image infos to .csv.

            Required argument:
            '-source' set path to directory containing .jpg images

            Optional argument:
            '-rgb'    calculate RGB mean values and standard deviations of all images

calc_rgb_stats() is based on the second solution from https://stackoverflow.com/a/73359050
"""

import argparse
import csv
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image


def check_images(img_list, save_path):
    """Check directory for corrupt .jpg images and save image dimensions to .csv."""
    img_info = []
    img_corrupt = []

    for image in img_list:
        try:
            with Image.open(image) as img:
                img_info.append({
                    "img_name": image.name,
                    "img_width": img.width,
                    "img_height": img.height
                })
        except Exception:
            img_info.append({
                "img_name": image.name,
                "img_width": "NA",
                "img_height": "NA"
            })
            img_corrupt.append(image)

    if img_corrupt:
        img_list = [image for image in img_list if image not in img_corrupt]
        corrupt_path = save_path / "images_corrupt"
        corrupt_path.mkdir(parents=True, exist_ok=True)
        for image in img_corrupt:
            Path(image).rename(corrupt_path / image.name)
        logging.info("Moved %s corrupt images to %s\n", len(img_corrupt), corrupt_path)

    with open(save_path / "images_info.csv", "w", newline="", encoding="utf-8") as info_file:
        writer_info = csv.DictWriter(info_file, fieldnames=img_info[0].keys())
        writer_info.writeheader()
        writer_info.writerows(img_info)

    return img_list, img_info


def calc_img_stats(img_info, save_path):
    """Calculate statistics of image dimensions and save result to .csv."""
    img_widths = np.array([info["img_width"] for info in img_info if info["img_width"] != "NA"])
    img_heights = np.array([info["img_height"] for info in img_info if info["img_height"] != "NA"])

    img_stats = [
        {"statistic": "mean",
         "img_width": round(np.mean(img_widths)),
         "img_height": round(np.mean(img_heights))},
        {"statistic": "median",
         "img_width": round(np.median(img_widths)),
         "img_height": round(np.median(img_heights))},
        {"statistic": "min",
         "img_width": np.min(img_widths),
         "img_height": np.min(img_heights)},
        {"statistic": "max",
         "img_width": np.max(img_widths),
         "img_height": np.max(img_heights)},
        {"statistic": "90th_percentile",
         "img_width": round(np.percentile(img_widths, 90)),
         "img_height": round(np.percentile(img_heights, 90))}
    ]

    with open(save_path / "images_stats.csv", "w", newline="", encoding="utf-8") as stats_file:
        writer_stats = csv.DictWriter(stats_file, fieldnames=img_stats[0].keys())
        writer_stats.writeheader()
        writer_stats.writerows(img_stats)


def calc_rgb_stats(img_list, save_path):
    """Calculate RGB mean values and standard deviations of images and save result to .csv."""
    rgb_values = np.concatenate([Image.open(image).getdata() for image in img_list], axis=0) / 255
    rgb_mean = np.around(np.mean(rgb_values, axis=0), 3)
    rgb_std = np.around(np.std(rgb_values, axis=0), 3)

    rgb_stats = [
        {"statistic": "mean",
         "R": rgb_mean[0],
         "G": rgb_mean[1],
         "B": rgb_mean[2]},
        {"statistic": "standard_deviation",
         "R": rgb_std[0],
         "G": rgb_std[1],
         "B": rgb_std[2]}
    ]

    with open(save_path / "images_rgb.csv", "w", newline="", encoding="utf-8") as rgb_file:
        writer_rgb = csv.DictWriter(rgb_file, fieldnames=rgb_stats[0].keys())
        writer_rgb.writeheader()
        writer_rgb.writerows(rgb_stats)


def main():
    """Get .jpg files in directory and check for corrupt images, write image infos to .csv."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-source", "--img_source", type=Path, required=True, metavar="PATH",
        help="Set path to directory containing .jpg images.")
    parser.add_argument("-rgb", "--rgb_mean_std", action="store_true",
        help="Calculate RGB mean values and standard deviations of all images.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    start_time = time.monotonic()

    img_source = args.img_source
    img_list = list(img_source.glob("**/*.jpg"))

    if img_list:
        logging.info("Found %s .jpg images in %s\n", len(img_list), img_source.resolve())
    else:
        logging.warning("Could not find any .jpg images in %s\n", img_source.resolve())
        sys.exit()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = img_source.parent / f"{timestamp}_images_processed"
    save_path.mkdir(parents=True, exist_ok=True)

    logging.info("Checking images...\n")
    img_list, img_info = check_images(img_list, save_path)
    calc_img_stats(img_info, save_path)

    if args.rgb_mean_std:
        logging.info("Calculating RGB mean values and standard deviations of all images...\n")
        calc_rgb_stats(img_list, save_path)

    logging.info("Information about processed images saved to: %s\n", save_path.resolve())
    logging.info("Script run time: %.2f s\n", time.monotonic() - start_time)


if __name__ == "__main__":
    main()
