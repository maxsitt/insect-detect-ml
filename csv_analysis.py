#!/usr/bin/env python3

'''
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Website:  https://maxsitt.github.io/insect-detect-docs/
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)

This Python script does the following:
- read "metadata_classified*.csv" file from current directory into pandas DataFrame
- calculate relative (or absolute, if frame width/height in mm is given) bounding box sizes
  and bbox length (= longer side) + bbox width (= shorter side)
- save metadata sorted by recording ID, tracking ID and timestamp successively (ascending)
  to "metadata_classified*_sorted.csv"

- group by tracking ID per recording ID and calculate per tracking ID (per recording ID):
  - number of images per top1 class
  - total number of images per tracking ID
  - mean classification probability for each top1 class
  - weighted mean classification probability for each top1 class
    (number images_top1 class / total number images_tracking ID) * mean classification probability
  - mean bbox length and bbox width for each top1 class
- save metadata sorted by recording ID, tracking ID and weighted probability successively
  (ascending) to "metadata_classified*_top1_all.csv"

- group by tracking ID per recording ID and calculate/extract per tracking ID (per recording ID):
  - mean detection confidence
  - first and last timestamp (start/end time)
  - duration [s] (end time - start time)
  - number of images of the top1 class with the highest weighted probability
  - name of the top1 class with the highest weighted probability
  - mean classification probability of the top1 class with the highest weighted probability
  - weighted classification probability of the top1 class with the highest weighted probability
  - mean bounding box length and width of the top1 class with the highest weighted probability
- remove tracking IDs with less or more than the specified number of images
- save metadata calculated for each tracking ID (per recording ID) (one row per tracking ID)
  to "metadata_classified*_top1.csv"

- extract some info about the analysis and save to "metadata_classified*_analysis_info.csv"
  - total number of unique tracking IDs per recording ID
  - minimum number of images per tracking ID to keep it in final .csv (*_top1.csv)
  - maximum number of images per tracking ID to keep it in final .csv (*_top1.csv)
  - number of unique trackings IDs per recording ID left after removing all
    tracking IDs with less or more than the specified number of images
  - number of unique top1 classes per recording ID after removing all
    tracking IDs with less or more than the specified number of images

- create and save some basic plots for a quick first data overview

- optional arguments:
  "-width [mm]" (default = 1) set absolute ("true") frame width in mm to calculate
                absolute bbox size [mm] (default of 1 gives relative bbox size)
  "-height [mm]" (default = 1) set absolute ("true") frame height in mm to calculate
                 absolute bbox size [mm] (default of 1 gives relative bbox size)
  "-min_tracks [number]" (default = 3) remove tracking IDs with less
                         than the specified number of images
  "-max_tracks [number]" (default = 1800) remove tracking IDs with more
                         than the specified number of images
'''

import argparse
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set start time for script execution timer
start_time = time.monotonic()

# Define optional arguments
parser = argparse.ArgumentParser()
parser.add_argument("-width", "--frame_width", type=int, default=1,
    help="absolute width [mm] of the frame during recording")
parser.add_argument("-height", "--frame_height", type=int, default=1,
    help="absolute height [mm] of the frame during recording")
parser.add_argument("-min_tracks", "--remove_min_tracks", type=int, default=3,
    help="remove tracking IDs with less than the specified number of images")
parser.add_argument("-max_tracks", "--remove_max_tracks", type=int, default=1800,
    help="remove tracking IDs with more than the specified number of images")
args = parser.parse_args()

# Get metadata (+ classification results) .csv file from current directory
# will always return the first metadata .csv file in alphabetical order
csv_file = Path(list(Path(".").glob("metadata_classified*.csv"))[0])

# Set name of the metadata .csv file (without extension)
csv_name = csv_file.stem

# Create folder to save analyzed metadata .csv files
timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S")
save_path = f"metadata_analyzed_{timestamp}"
Path(f"{save_path}/{csv_name}_tables").mkdir(parents=True, exist_ok=True)

# Read metadata .csv into pandas DataFrame and convert timestamp to pandas datetime object
df = pd.read_csv(csv_file, encoding="utf-8")
df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y%m%d_%H-%M-%S.%f")

# Calculate bounding box sizes
# if args.frame_width/_height == 1 (default): relative bbox size
df["bbox_size_x"] = round((df["x_max"] - df["x_min"]) * args.frame_width, 4)
df["bbox_size_y"] = round((df["y_max"] - df["y_min"]) * args.frame_height, 4)

# Get bbox length (= longer side) and bbox width (= shorter side)
df["bbox_length"] = np.where(df["bbox_size_x"] > df["bbox_size_y"], df["bbox_size_x"], df["bbox_size_y"])
df["bbox_width"] = np.where(df["bbox_size_x"] < df["bbox_size_y"], df["bbox_size_x"], df["bbox_size_y"])

# Sort metadata by recording ID, tracking ID and timestamp successively (ascending)
df = df.sort_values(["rec_ID", "track_ID", "timestamp"])

# Write DataFrame to csv file
df.to_csv(f"{save_path}/{csv_name}_tables/{csv_name}_sorted.csv", index=False)

# Create DataFrame with recording ID, tracking ID and top1 classes + number of corresponding images per tracking ID
df_top1_all = (df.groupby(["rec_ID", "track_ID"])["top1"]
                 .value_counts(sort=False)
                 .reset_index(name="top1_imgs"))

# Sum the images of all top1 classes per tracking ID
df_top1_all["track_ID_imgs"] = (df_top1_all.groupby(["rec_ID", "track_ID"])["top1_imgs"]
                                           .transform("sum")
                                           .reset_index(drop=True))

# Calculate mean classification probability for each top1 class per tracking ID
df_top1_all["prob_mean"] = (df.groupby(["rec_ID", "track_ID", "top1"])["top1_prob"]
                              .mean()
                              .round(2)
                              .reset_index(drop=True))

# Calculate weighted mean classification probability for each top1 class per tracking ID
df_top1_all["prob_weighted"] = (((df_top1_all["top1_imgs"] / df_top1_all["track_ID_imgs"]) * df_top1_all["prob_mean"])
                                  .round(2))

# Calculate mean bounding box length and width for each top1 class per tracking ID
df_top1_all["bbox_length_mean"] = (df.groupby(["rec_ID", "track_ID", "top1"])["bbox_length"]
                                     .mean()
                                     .round(3)
                                     .reset_index(drop=True))
df_top1_all["bbox_width_mean"] = (df.groupby(["rec_ID", "track_ID", "top1"])["bbox_width"]
                                    .mean()
                                    .round(3)
                                    .reset_index(drop=True))

# Sort data by recording ID, tracking ID and weighted probability successively (ascending)
df_top1_all = df_top1_all.sort_values(["rec_ID", "track_ID", "prob_weighted"])

# Write DataFrame to csv file
df_top1_all.to_csv(f"{save_path}/{csv_name}_tables/{csv_name}_top1_all.csv", index=False)

# Create DataFrame with recording ID and tracking ID + number of corresponding images
df_top1 = (df.groupby(["rec_ID", "track_ID"])["track_ID"]
             .count()
             .reset_index(name="track_ID_imgs"))

# Calculate mean detection confidence per tracking ID
df_top1["det_conf_mean"] = (df.groupby(["rec_ID", "track_ID"])["confidence"]
                              .mean()
                              .round(2)
                              .reset_index(drop=True))

# Extract date of each tracking ID from first timestamp
df_top1["date"] = (df.groupby(["rec_ID", "track_ID"])["timestamp"]
                     .min().dt.date
                     .reset_index(drop=True))

# Get start and end time of each tracking ID from first and last timestamp
df_top1["start_time"] = (df.groupby(["rec_ID", "track_ID"])["timestamp"]
                           .min()
                           .reset_index(drop=True))
df_top1["end_time"] = (df.groupby(["rec_ID", "track_ID"])["timestamp"]
                         .max()
                         .reset_index(drop=True))

# Calculate duration for each tracking ID in seconds
df_top1["duration_s"] = (df_top1["end_time"] - df_top1["start_time"]) / pd.Timedelta(seconds=1)

# Get number of images, name, mean + weighted classification probability of the top1 class with the highest weighted probability
df_top1["top1_imgs"] = (df_top1_all.loc[df_top1_all
                                   .groupby(["rec_ID", "track_ID"])["prob_weighted"]
                                   .idxmax()]["top1_imgs"]
                                   .reset_index(drop=True))
df_top1["top1_class"] = (df_top1_all.loc[df_top1_all
                                    .groupby(["rec_ID", "track_ID"])["prob_weighted"]
                                    .idxmax()]["top1"]
                                    .reset_index(drop=True))
df_top1["top1_prob_mean"] = (df_top1_all.loc[df_top1_all
                                        .groupby(["rec_ID", "track_ID"])["prob_weighted"]
                                        .idxmax()]["prob_mean"]
                                        .reset_index(drop=True))
df_top1["top1_prob_weighted"] = (df_top1_all.groupby(["rec_ID", "track_ID"])["prob_weighted"]
                                            .max()
                                            .reset_index(drop=True))

# Get mean bounding box length and width of the top1 class with the highest weighted probability
df_top1["bbox_length_mean"] = (df_top1_all.loc[df_top1_all
                                          .groupby(["rec_ID", "track_ID"])["prob_weighted"]
                                          .idxmax()]["bbox_length_mean"]
                                          .reset_index(drop=True))
df_top1["bbox_width_mean"] = (df_top1_all.loc[df_top1_all
                                         .groupby(["rec_ID", "track_ID"])["prob_weighted"]
                                         .idxmax()]["bbox_width_mean"]
                                         .reset_index(drop=True))

# Remove tracking IDs with less (default: 3) or more (default: 1800) than the specified number of images
removed_min_tracks = len(df_top1[df_top1["track_ID_imgs"] < args.remove_min_tracks])
removed_max_tracks = len(df_top1[df_top1["track_ID_imgs"] > args.remove_max_tracks])
print(f"\nRemoved {removed_min_tracks} tracking IDs with less than {args.remove_min_tracks} images.")
print(f"Removed {removed_max_tracks} tracking IDs with more than {args.remove_max_tracks} images.\n")
df_top1 = df_top1[(df_top1["track_ID_imgs"] >= args.remove_min_tracks) &
                  (df_top1["track_ID_imgs"] <= args.remove_max_tracks)]

# Write DataFrame to csv file
df_top1.to_csv(f"{save_path}/{csv_name}_tables/{csv_name}_top1.csv", index=False)

# Create DataFrame with info about the analysis
df_info = (df.groupby("rec_ID", as_index=True)
             .agg(track_IDs_total=("track_ID", "nunique"),
                  top1_classes_total=("top1", "nunique")))
df_info["min_track_imgs"] = args.remove_min_tracks
df_info["max_track_imgs"] = args.remove_max_tracks
df_info["track_IDs_left"] = (df_top1.groupby("rec_ID")
                                    .size())
df_info["top1_classes_left"] = (df_top1.groupby("rec_ID")["top1_class"]
                                       .nunique())

# Write DataFrame to csv file
df_info.to_csv(f"{save_path}/{csv_name}_analysis_info.csv")

# Create folder to save plots
Path(f"{save_path}/{csv_name}_plots").mkdir(parents=True, exist_ok=True)

# Plot histogram with the distribution of the number of images per tracking ID
(df_top1["track_ID_imgs"].plot(kind="hist",
                               bins=range(min(df_top1["track_ID_imgs"]), max(df_top1["track_ID_imgs"]) + 1, 1),
                               edgecolor="black",
                               title="Distribution of the number of images per tracking ID")
                         .set(xlabel="Number of images per tracking ID"))
plt.savefig(f"{save_path}/{csv_name}_plots/imgs_per_track.png", bbox_inches="tight")
plt.clf()

# Plot histogram with the distribution of the duration [s] per tracking ID
(df_top1["duration_s"].plot(kind="hist",
                            bins=np.arange(min(df_top1["duration_s"]), max(df_top1["duration_s"]) + 1, 1),
                            edgecolor="black",
                            title="Distribution of the duration [s] per tracking ID")
                      .set(xlabel="Duration [s] per tracking ID"))
plt.savefig(f"{save_path}/{csv_name}_plots/duration_per_track.png", bbox_inches="tight")
plt.clf()

# Plot barplot with the number of individual tracking IDs per recording
(df_top1.groupby(["date", "rec_ID"])["track_ID"]
        .nunique()
        .plot(kind="bar",
              edgecolor="black",
              ylabel="Number of individual tracking IDs",
              title="Number of individual tracking IDs per recording"))
plt.savefig(f"{save_path}/{csv_name}_plots/tracks_per_rec.png", bbox_inches="tight")
plt.clf()

# Plot stacked barplot with the top 1 classes per recording
(df_top1.groupby(["date", "rec_ID"])["top1_class"]
        .value_counts()
        .unstack()
        .plot(kind="bar",
              stacked=True,
              edgecolor="black",
              ylabel="Number of individual tracking IDs",
              title="Top 1 classes per recording"))
plt.savefig(f"{save_path}/{csv_name}_plots/top1_classes_per_rec.png", bbox_inches="tight")
plt.clf()

# Plot barplot with the mean detection confidence per top 1 class
(df_top1.groupby(["top1_class"])["det_conf_mean"]
        .mean()
        .sort_values(ascending=False)
        .plot(kind="bar",
              edgecolor="black",
              ylabel="Mean detection confidence",
              title="Mean detection confidence per top 1 class"))
plt.savefig(f"{save_path}/{csv_name}_plots/top1_classes_mean_det_conf.png", bbox_inches="tight")
plt.clf()

# Plot barplot with the mean classification probability per top 1 class
(df_top1.groupby(["top1_class"])["top1_prob_mean"]
        .mean()
        .sort_values(ascending=False)
        .plot(kind="bar",
              edgecolor="black",
              ylabel="Mean classification probability",
              title="Mean classification probability per top 1 class"))
plt.savefig(f"{save_path}/{csv_name}_plots/top1_classes_mean_prob.png", bbox_inches="tight")
plt.clf()

# Plot barplot with the mean tracking duration [s] per top 1 class
(df_top1.groupby(["top1_class"])["duration_s"]
        .mean()
        .sort_values(ascending=False)
        .plot(kind="bar",
              edgecolor="black",
              ylabel="Mean tracking duration [s]",
              title="Mean tracking duration [s] per top 1 class"))
plt.savefig(f"{save_path}/{csv_name}_plots/top1_classes_mean_duration.png", bbox_inches="tight")
plt.clf()

# Plot barplot with the mean number of images per tracking ID classified as top 1 class
(df_top1.groupby(["top1_class"])["top1_imgs"]
        .mean()
        .sort_values(ascending=False)
        .plot(kind="bar",
              edgecolor="black",
              ylabel="Mean number of images per tracking ID",
              title="Mean number of images per tracking ID classified as top 1 class"))
plt.savefig(f"{save_path}/{csv_name}_plots/top1_classes_mean_imgs.png", bbox_inches="tight")
plt.clf()

# Plot boxplot with mean bbox lengths per top 1 class
Y_LABEL_L = "Mean relative bounding box length"
if args.frame_width != 1 or args.frame_height != 1:
    Y_LABEL_L = "Mean bounding box length [mm]"
(df_top1.plot(kind="box",
              column="bbox_length_mean",
              by="top1_class",
              ylabel=Y_LABEL_L,
              title="Mean bounding box length (longer side) per top 1 class"))
plt.savefig(f"{save_path}/{csv_name}_plots/top1_classes_bbox_length.png", bbox_inches="tight")
plt.clf()

# Plot boxplot with mean bbox widths per top 1 class
Y_LABEL_W = "Mean relative bounding box width"
if args.frame_width != 1 or args.frame_height != 1:
    Y_LABEL_W = "Mean bounding box width [mm]"
(df_top1.plot(kind="box",
              column="bbox_width_mean",
              by="top1_class",
              ylabel=Y_LABEL_W,
              title="Mean bounding box width (shorter side) per top 1 class"))
plt.savefig(f"{save_path}/{csv_name}_plots/top1_classes_bbox_width.png", bbox_inches="tight")
plt.clf()

# Print info messages to console
path_tables = Path.cwd() / f"{save_path}/{csv_name}_tables"
path_plots = Path.cwd() / f"{save_path}/{csv_name}_plots"
print(f"Generated .csv files saved to: {path_tables}")
print(f"Generated plots saved to:      {path_plots}\n")
print(f"Script run time: {round(time.monotonic() - start_time, 3)} s")
