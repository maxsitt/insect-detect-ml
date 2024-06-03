"""Functions to read, post-process and analyze a metadata .csv file with classification results.

Source:   https://github.com/maxsitt/insect-detect-ml
License:  GNU AGPLv3 (https://choosealicense.com/licenses/agpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

Functions:
    read_sort_metadata(): Read metadata .csv file and save sorted metadata to .csv.
    process_track_id(): Process metadata for each top1 class per tracking ID per recording ID
                        and save result to .csv.
    process_top1_class(): Process metadata for the top1 class with the highest weighted probability
                          and save result to .csv.
    processing_log(): Get information about the post-processing run and save to .csv.
    create_plots(): Create overview plots with the post-processed metadata and save to .png.

    main(): Read and post-process metadata, save results together with info log and create plots.

            Required argument:
            '-source'   set path to directory containing metadata .csv with classification results
                        -> will get the last metadata .csv file in alphabetical order

            Optional arguments:
            '-size'     set absolute frame width and height in mm to calculate 'true' bbox size [mm]
                        -> e.g. '-size 350 200' for small platform (350x200 mm)
                        -> default of (1, 1) gives relative bbox size

            mutually exclusive - use only one of the following arguments:
            '-images'   remove tracking IDs with less or more than the specified number of images
                        -> e.g. '-images 3 1800' to remove tracking IDs
                           with less than 3 or more than 1800 images
            OR
            '-duration' remove tracking IDs with less or more than the specified duration in seconds
                        -> e.g. '-duration 2 1800' to remove tracking IDs that
                           were tracked less than 2 or more than 1800 seconds
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import distinctipy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_sort_metadata(csv_file, csv_name, save_path, frame_width=1, frame_height=1):
    """Read metadata .csv file and save sorted metadata to .csv."""
    df = pd.read_csv(csv_file, encoding="utf-8", parse_dates=["timestamp"])
    if df["timestamp"].dtypes != "datetime64[ns]":
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y%m%d_%H-%M-%S.%f")  # old format

    df["bbox_size_x"] = round((df["x_max"] - df["x_min"]) * frame_width, 4)
    df["bbox_size_y"] = round((df["y_max"] - df["y_min"]) * frame_height, 4)
    df["bbox_length"] = np.maximum(df["bbox_size_x"], df["bbox_size_y"])
    df["bbox_width"] = np.minimum(df["bbox_size_x"], df["bbox_size_y"])

    df.sort_values(["cam_ID", "rec_ID", "track_ID", "timestamp"], inplace=True)
    df.to_csv(save_path / f"{csv_name}_sorted.csv", index=False)

    return df


def process_track_id(df, csv_name, save_path):
    """Process metadata for each top1 class per tracking ID per recording ID and save result to .csv.

    Calculate number of images for each top1 class and total number of images per tracking ID,
    mean + weighted classification probability and mean bounding box length + width
    for each top1 class per tracking ID per recording ID.
    """
    df_grouped_top1 = df.groupby(["cam_ID", "rec_ID", "track_ID", "top1"])
    df_top1_all = df_grouped_top1.size().reset_index(name="top1_imgs")

    df_top1_all["track_ID_imgs"] = df_top1_all.groupby(["cam_ID", "rec_ID", "track_ID"])["top1_imgs"].transform("sum")

    df_top1_all["top1_prob_mean"] = df_grouped_top1["top1_prob"].mean().round(2).reset_index(drop=True)
    df_top1_all["top1_prob_weighted"] = (df_top1_all["top1_prob_mean"] * (df_top1_all["top1_imgs"] / df_top1_all["track_ID_imgs"])).round(2)

    df_top1_all["bbox_length_mean"] = df_grouped_top1["bbox_length"].mean().round(3).reset_index(drop=True)
    df_top1_all["bbox_width_mean"] = df_grouped_top1["bbox_width"].mean().round(3).reset_index(drop=True)

    df_top1_all.sort_values(["cam_ID", "rec_ID", "track_ID", "top1_prob_weighted"], ascending=[True, True, True, False], inplace=True)
    df_top1_all = df_top1_all[["cam_ID", "rec_ID", "track_ID", "track_ID_imgs", "top1_imgs", "top1",
                               "top1_prob_mean", "top1_prob_weighted", "bbox_length_mean", "bbox_width_mean"]]
    df_top1_all.to_csv(save_path / f"{csv_name}_top1_all.csv", index=False)

    return df_top1_all


def process_top1_class(df, df_top1_all, csv_name, save_path, images=None, duration=None):
    """Process metadata for the top1 class with the highest weighted probability and save result to .csv.

    Get date, start + end time and calculate duration [s] and mean detection confidence per tracking ID.
    Get number of images, class name, mean + weighted classification probability and mean bounding
    box length + width for the top1 class with the highest weighted probability per tracking ID.
    Optionally remove tracking IDs with less or more than the specified number of images or duration.
    """
    df_grouped_trackid = df.groupby(["cam_ID", "rec_ID", "track_ID"])
    df_top1_final = df_grouped_trackid.size().reset_index(name="track_ID_imgs")

    timestamps = df_grouped_trackid["timestamp"].agg(["min", "max"])
    df_top1_final["date"] = timestamps["min"].dt.date.reset_index(drop=True)
    df_top1_final["start_time"] = timestamps["min"].reset_index(drop=True)
    df_top1_final["end_time"] = timestamps["max"].reset_index(drop=True)
    df_top1_final["duration_s"] = ((df_top1_final["end_time"] - df_top1_final["start_time"]) / pd.Timedelta(seconds=1)).round(2)
    df_top1_final["det_conf_mean"] = df_grouped_trackid["confidence"].mean().round(2).reset_index(drop=True)

    prob_idxmax = df_top1_all.groupby(["cam_ID", "rec_ID", "track_ID"])["top1_prob_weighted"].idxmax()
    df_top1_final = df_top1_final.join(df_top1_all.loc[prob_idxmax, ["top1_imgs", "top1", "top1_prob_mean",
                                                                     "top1_prob_weighted", "bbox_length_mean",
                                                                     "bbox_width_mean"]].reset_index(drop=True))

    if images:
        tracks_images = df_top1_final["track_ID_imgs"]
        tracks_min_images = (tracks_images < images[0]).sum()
        tracks_max_images = (tracks_images > images[1]).sum()
        df_top1_final = df_top1_final[tracks_images.between(images[0], images[1])]
        logging.info("Removed %s tracking IDs with less than %s images.", tracks_min_images, images[0])
        logging.info("Removed %s tracking IDs with more than %s images.\n", tracks_max_images, images[1])
    elif duration:
        tracks_duration = df_top1_final["duration_s"]
        tracks_min_duration = (tracks_duration < duration[0]).sum()
        tracks_max_duration = (tracks_duration > duration[1]).sum()
        df_top1_final = df_top1_final[tracks_duration.between(duration[0], duration[1])]
        logging.info("Removed %s tracking IDs with less than %s seconds duration.", tracks_min_duration, duration[0])
        logging.info("Removed %s tracking IDs with more than %s seconds duration.\n", tracks_max_duration, duration[1])
    else:
        logging.info("No tracking IDs were removed. It is recommended to remove tracking IDs "
                     "with less than 3 images or which were tracked less than 2 seconds.\n"
                     "Usage:    process_metadata.py -source PATH -images 3 1800\n"
                     "      OR: process_metadata.py -source PATH -duration 2 1800\n")

    df_top1_final = df_top1_final[["cam_ID", "rec_ID", "date", "track_ID", "start_time", "end_time", "duration_s",
                                   "det_conf_mean", "track_ID_imgs", "top1_imgs", "top1", "top1_prob_mean",
                                   "top1_prob_weighted", "bbox_length_mean", "bbox_width_mean"]]
    df_top1_final.to_csv(save_path / f"{csv_name}_top1_final.csv", index=False)

    return df_top1_final


def processing_log(df, df_top1_final, csv_name, save_path, images=None, duration=None):
    """Get information about the post-processing run and save to .csv."""
    df_info = df.groupby("rec_ID", as_index=True).agg(track_IDs_total=("track_ID", "nunique"),
                                                      top1_classes_total=("top1", "nunique"))

    df_top1_final_grouped = df_top1_final.groupby("rec_ID")
    df_info["track_IDs_left"] = df_top1_final_grouped.size().fillna(0).astype(int)
    df_info["top1_classes_left"] = df_top1_final_grouped["top1"].nunique().fillna(0).astype(int)

    if images:
        df_info["track_min_images"] = images[0]
        df_info["track_max_images"] = images[1]
    elif duration:
        df_info["track_min_duration"] = duration[0]
        df_info["track_max_duration"] = duration[1]

    df_info.to_csv(save_path / f"{csv_name}_processing_log.csv")


def create_plots(df_top1_final, save_path, frame_width=1, frame_height=1):
    """Create overview plots with the post-processed metadata and save to .png."""
    # Draw grid lines behind other elements
    plt.rcParams["axes.axisbelow"] = True

    # Plot histogram with the distribution of the recording duration per tracking ID
    duration_min = min(df_top1_final["duration_s"])
    duration_max = max(df_top1_final["duration_s"])
    if duration_min == duration_max:
        duration_max += 1
    (df_top1_final["duration_s"].plot(kind="hist",
                                      figsize=(15, 10),
                                      bins=np.arange(duration_min, duration_max + 1, 1),
                                      title="Distribution of the recording duration per tracking ID")
                                .set(xlabel="Recording duration per tracking ID [s]"))
    plt.grid(axis="y", color="gray", linewidth=0.5, alpha=0.2)
    plt.savefig(save_path / "track_duration.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot histogram with the distribution of the number of images per tracking ID
    images_min = min(df_top1_final["track_ID_imgs"])
    images_max = max(df_top1_final["track_ID_imgs"])
    if images_min == images_max:
        images_max += 1
    (df_top1_final["track_ID_imgs"].plot(kind="hist",
                                         figsize=(15, 10),
                                         bins=range(images_min, images_max + 1, 1),
                                         title="Distribution of the number of images per tracking ID")
                                   .set(xlabel="Number of images per tracking ID"))
    plt.grid(axis="y", color="gray", linewidth=0.5, alpha=0.2)
    plt.savefig(save_path / "track_images.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot boxplot with mean bbox lengths per top 1 class
    if frame_width != 1 or frame_height != 1:
        y_label_l = "Mean bounding box length [mm]"
    else:
        y_label_l = "Mean relative bounding box length"
    (df_top1_final.plot(kind="box",
                        column="bbox_length_mean",
                        by="top1",
                        rot=90,
                        figsize=(15, 10),
                        xlabel="Top 1 class",
                        ylabel=y_label_l))
    plt.grid(axis="y", color="gray", linewidth=0.5, alpha=0.2)
    plt.suptitle("")
    plt.title("Mean bounding box length (longer side) per top 1 class")
    plt.savefig(save_path / "top1_bbox_length.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot boxplot with mean bbox widths per top 1 class
    if frame_width != 1 or frame_height != 1:
        y_label_w = "Mean bounding box width [mm]"
    else:
        y_label_w = "Mean relative bounding box width"
    (df_top1_final.plot(kind="box",
                        column="bbox_width_mean",
                        by="top1",
                        rot=90,
                        figsize=(15, 10),
                        xlabel="Top 1 class",
                        ylabel=y_label_w))
    plt.grid(axis="y", color="gray", linewidth=0.5, alpha=0.2)
    plt.suptitle("")
    plt.title("Mean bounding box width (shorter side) per top 1 class")
    plt.savefig(save_path / "top1_bbox_width.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot barplot with the mean detection confidence per top 1 class
    (df_top1_final.groupby(["top1"])["det_conf_mean"]
                  .mean()
                  .sort_values(ascending=False)
                  .plot(kind="bar",
                        edgecolor="black",
                        rot=90,
                        ylim=(0,1),
                        yticks=([x/10 for x in range(0, 11)]),
                        figsize=(15, 10),
                        xlabel="Top 1 class",
                        ylabel="Mean detection confidence",
                        title="Mean detection confidence per top 1 class"))
    plt.grid(axis="y", color="gray", linewidth=0.5, alpha=0.2)
    plt.savefig(save_path / "top1_mean_det_conf.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot barplot with the mean tracking duration per top 1 class
    (df_top1_final.groupby(["top1"])["duration_s"]
                  .mean()
                  .sort_values(ascending=False)
                  .plot(kind="bar",
                        figsize=(15, 10),
                        edgecolor="black",
                        xlabel="Top 1 class",
                        ylabel="Mean tracking duration [s]",
                        title="Mean tracking duration per top 1 class"))
    plt.grid(axis="y", color="gray", linewidth=0.5, alpha=0.2)
    plt.savefig(save_path / "top1_mean_duration.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot barplot with the mean number of images per tracking ID classified as top 1 class
    (df_top1_final.groupby(["top1"])["top1_imgs"]
                  .mean()
                  .sort_values(ascending=False)
                  .plot(kind="bar",
                        figsize=(15, 10),
                        edgecolor="black",
                        xlabel="Top 1 class",
                        ylabel="Mean number of images per tracking ID",
                        title="Mean number of images per tracking ID classified as top 1 class"))
    plt.grid(axis="y", color="gray", linewidth=0.5, alpha=0.2)
    plt.savefig(save_path / "top1_mean_images.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot boxplot with the classification probability per top 1 class
    (df_top1_final.plot(kind="box",
                        column="top1_prob_mean",
                        by="top1",
                        rot=90,
                        ylim=(0,1),
                        yticks=([x/10 for x in range(0, 11)]),
                        figsize=(15, 10),
                        xlabel="Top 1 class",
                        ylabel="Classification probability"))
    plt.grid(axis="y", color="gray", linewidth=0.5, alpha=0.2)
    plt.suptitle("")
    plt.title("Classification probability per top 1 class")
    plt.savefig(save_path / "top1_prob.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot barplot with the mean classification probability per top 1 class
    (df_top1_final.groupby(["top1"])["top1_prob_mean"]
                  .mean()
                  .sort_values(ascending=False)
                  .plot(kind="bar",
                        edgecolor="black",
                        rot=90,
                        ylim=(0,1),
                        yticks=([x/10 for x in range(0, 11)]),
                        figsize=(15, 10),
                        xlabel="Top 1 class",
                        ylabel="Mean classification probability",
                        title="Mean classification probability per top 1 class"))
    plt.grid(axis="y", color="gray", linewidth=0.5, alpha=0.2)
    plt.savefig(save_path / "top1_prob_mean.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot barplot with the number of individual tracking IDs per recording
    (df_top1_final.groupby(["date", "rec_ID"])["track_ID"]
                  .nunique()
                  .plot(kind="bar",
                        figsize=(15, 10),
                        edgecolor="black",
                        xlabel="Recording interval [Date, Rec ID]",
                        ylabel="Number of individual tracking IDs",
                        title="Number of individual tracking IDs per recording"))
    plt.grid(axis="y", color="gray", linewidth=0.5, alpha=0.2)
    plt.xticks(fontsize="xx-small")
    plt.savefig(save_path / "rec_id_tracks.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Create custom colormap with distinctipy
    num_top1 = df_top1_final["top1"].nunique()
    colors = distinctipy.get_colors(num_top1)
    cmap_custom = distinctipy.get_colormap(colors)

    # Plot stacked barplot with the top 1 classes per recording
    (df_top1_final.groupby(["date", "rec_ID"])["top1"]
                  .value_counts()
                  .unstack()
                  .plot(kind="bar",
                        figsize=(15,10),
                        stacked=True,
                        colormap=cmap_custom,
                        edgecolor="black",
                        xlabel="Recording interval [Date, Rec ID]",
                        ylabel="Number of individual tracking IDs",
                        title="Top 1 classes per recording"))
    plt.grid(axis="y", color="gray", linewidth=0.5, alpha=0.2)
    plt.xticks(fontsize="xx-small")
    plt.savefig(save_path / "rec_id_top1.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """Read and post-process metadata, save results together with info log and create plots."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-source", "--csv_source", type=Path, required=True, metavar="PATH",
        help="Set path to directory containing metadata .csv file(s) with classification results.")
    parser.add_argument("-size", "--frame_size", nargs=2, type=int, default=(1, 1), metavar=("WIDTH", "HEIGHT"),
        help="Set absolute frame width and height in millimeters (default: (1, 1) for relative frame size).")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-images", "--track_images", nargs=2, type=int, metavar=("MIN", "MAX"),
        help="Remove tracking IDs with less or more than the specified number of images.")
    group.add_argument("-duration", "--track_duration", nargs=2, type=int, metavar=("MIN", "MAX"),
        help="Remove tracking IDs with less or more than the specified duration in seconds.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    start_time = time.monotonic()

    csv_source = args.csv_source
    csv_list = list(csv_source.glob("*metadata_classified*.csv"))

    if csv_list:
        logging.info("Found %s metadata .csv file(s) in %s\n", len(csv_list), csv_source.resolve())
        csv_file = csv_list[-1]
        csv_name = csv_file.stem
    else:
        logging.warning("Could not find any metadata .csv file(s) in %s\n", csv_source.resolve())
        sys.exit()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = csv_file.parent / f"{timestamp}_metadata_processed"
    save_path.mkdir(parents=True, exist_ok=True)

    logging.info("Processing metadata...\n")
    df = read_sort_metadata(csv_file, csv_name, save_path, args.frame_size[0], args.frame_size[1])
    df_top1_all = process_track_id(df, csv_name, save_path)
    df_top1_final = process_top1_class(df, df_top1_all, csv_name, save_path, args.track_images, args.track_duration)

    processing_log(df, df_top1_final, csv_name, save_path, args.track_images, args.track_duration)

    logging.info("Creating plots...\n")
    create_plots(df_top1_final, save_path, args.frame_size[0], args.frame_size[1])

    logging.info("Post-processed metadata and plots saved to: %s\n", save_path.resolve())
    logging.info("Script run time: %.2f s\n", time.monotonic() - start_time)


if __name__ == "__main__":
    main()
