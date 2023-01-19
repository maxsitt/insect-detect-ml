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
  as "metadata_classified*_sorted.csv"

- group by tracking ID per recording ID and calculate per tracking ID (per recording ID):
  - number of images per top1 class
  - total number of images per tracking ID
  - mean classification probability for each top1 class
  - weighted mean classification probability for each top1 class
    (number images_top1 class / total number images_tracking ID) * mean classification probability
  - mean bbox length and bbox width for each top1 class
- save metadata sorted by recording ID, tracking ID and weighted probability successively
  (ascending) as "metadata_classified*_top1_all.csv"

- group by tracking ID per recording ID and calculate/extract per tracking ID (per recording ID):
  - mean detection confidence
  - first and last timestamp (start/end time)
  - duration [s] (end time - start time)
  - number of images of the top1 class with the highest weighted probability
  - name of the top1 class with the highest weighted probability
  - mean classification probability of the top1 class with the highest weighted probability
  - weighted classification probability of the top1 class with the highest weighted probability
  - mean bounding box length and width of the top1 class with the highest weighted probability
- save metadata calculated for each tracking ID (per recording ID) (one row per tracking ID)
  as "metadata_classified*_top1.csv"

- create and save some basic plots for a quick first data overview
'''

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set start time for script execution timer
start_time = time.monotonic()

# Get metadata (+ classification results) .csv file from current directory
# will always return the first metadata .csv file in alphabetical order
csv_file = Path(list(Path(".").glob("metadata_classified*.csv"))[0])

# Set name of the metadata .csv file (without extension)
csv_name = csv_file.stem

# Create folder to save analyzed metadata .csv files
Path(f"{csv_name}_tables").mkdir(parents=True, exist_ok=True)

# Read metadata .csv into pandas DataFrame and convert timestamp to pandas datetime object
df = pd.read_csv(csv_file, encoding="utf-8")
df["timestamp"] = pd.to_datetime(df.timestamp, format="%Y%m%d_%H-%M-%S.%f")

# Calculate bounding box sizes
# if FRAME_WIDTH/_HEIGHT == 1: relative bbox size
# insert true frame width/height in mm to calculate absolute bbox size [mm]
FRAME_WIDTH = 1
FRAME_HEIGHT = 1
df["bbox_size_x"] = round((df.x_max - df.x_min) * FRAME_WIDTH, 4)
df["bbox_size_y"] = round((df.y_max - df.y_min) * FRAME_HEIGHT, 4)

# Get bbox length (= longer side) and bbox width (= shorter side)
df["bbox_length"] = np.where(df["bbox_size_x"] > df["bbox_size_y"], df["bbox_size_x"], df["bbox_size_y"])
df["bbox_width"] = np.where(df["bbox_size_x"] < df["bbox_size_y"], df["bbox_size_x"], df["bbox_size_y"])

# Sort metadata by recording ID, tracking ID and timestamp successively (ascending)
df = df.sort_values(["rec_ID", "track_ID", "timestamp"])

# Write DataFrame to csv file
df.to_csv(f"{csv_name}_tables/{csv_name}_sorted.csv", index=False)


# Create DataFrame with recording ID, tracking ID and top1 classes + number of corresponding images per tracking ID
df_top1_all = df.groupby(["rec_ID", "track_ID"])["top1"].value_counts(sort=False).reset_index(name="top1_imgs")

# Sum the images of all top1 classes per tracking ID
df_top1_all["track_ID_imgs"] = df_top1_all.groupby(["rec_ID", "track_ID"])["top1_imgs"].transform("sum").reset_index(drop=True)

# Calculate mean classification probability for each top1 class per tracking ID
df_top1_all["prob_mean"] = df.groupby(["rec_ID", "track_ID", "top1"])["top1_prob"].mean().round(2).reset_index(drop=True)

# Calculate weighted mean classification probability for each top1 class per tracking ID
df_top1_all["prob_weighted"] = ((df_top1_all.top1_imgs / df_top1_all.track_ID_imgs) * df_top1_all.prob_mean).round(3)

# Calculate mean bounding box length and width for each top1 class per tracking ID
df_top1_all["bbox_length_mean"] = df.groupby(["rec_ID", "track_ID", "top1"])["bbox_length"].mean().round(4).reset_index(drop=True)
df_top1_all["bbox_width_mean"] = df.groupby(["rec_ID", "track_ID", "top1"])["bbox_width"].mean().round(4).reset_index(drop=True)

# Sort data by recording ID, tracking ID and weighted probability successively (ascending)
df_top1_all = df_top1_all.sort_values(["rec_ID", "track_ID", "prob_weighted"])

# Write DataFrame to csv file
df_top1_all.to_csv(f"{csv_name}_tables/{csv_name}_top1_all.csv", index=False)


# Create DataFrame with recording ID and tracking ID + number of corresponding images
df_top1 = df.groupby(["rec_ID", "track_ID"])["track_ID"].count().reset_index(name="track_ID_imgs")

# Calculate mean detection confidence per tracking ID
df_top1["det_conf_mean"] = df.groupby(["rec_ID", "track_ID"])["confidence"].mean().round(2).reset_index(drop=True)

# Extract date of each tracking ID from first timestamp
df_top1["date"] = df.groupby(["rec_ID", "track_ID"])["timestamp"].min().dt.date.reset_index(drop=True)

# Get start and end time of each tracking ID from first and last timestamp
df_top1["start_time"] = df.groupby(["rec_ID", "track_ID"])["timestamp"].min().reset_index(drop=True)
df_top1["end_time"] = df.groupby(["rec_ID", "track_ID"])["timestamp"].max().reset_index(drop=True)

# Calculate duration for each tracking ID in seconds
df_top1["duration_s"] = (df_top1.end_time - df_top1.start_time) / pd.Timedelta(seconds=1)

# Get number of images, name, mean + weighted classification probability of the top1 class with the highest weighted probability
df_top1["top1_imgs"] = df_top1_all.loc[df_top1_all.groupby(["rec_ID", "track_ID"])["prob_weighted"].idxmax()]["top1_imgs"].reset_index(drop=True)
df_top1["top1_class"] = df_top1_all.loc[df_top1_all.groupby(["rec_ID", "track_ID"])["prob_weighted"].idxmax()]["top1"].reset_index(drop=True)
df_top1["top1_prob_mean"] = df_top1_all.loc[df_top1_all.groupby(["rec_ID", "track_ID"])["prob_weighted"].idxmax()]["prob_mean"].reset_index(drop=True)
df_top1["top1_prob_weighted"] = df_top1_all.groupby(["rec_ID", "track_ID"])["prob_weighted"].max().reset_index(drop=True)

# Get mean bounding box length and width of the top1 class with the highest weighted probability
df_top1["bbox_length_mean"] = df_top1_all.loc[df_top1_all.groupby(["rec_ID", "track_ID"])["prob_weighted"].idxmax()]["bbox_length_mean"].reset_index(drop=True)
df_top1["bbox_width_mean"] = df_top1_all.loc[df_top1_all.groupby(["rec_ID", "track_ID"])["prob_weighted"].idxmax()]["bbox_width_mean"].reset_index(drop=True)

# Write DataFrame to csv file
df_top1.to_csv(f"{csv_name}_tables/{csv_name}_top1.csv", index=False)


# Create folder to save plots
Path(f"{csv_name}_plots").mkdir(parents=True, exist_ok=True)

# Plot histogram with the distribution of the number of images per tracking ID
(df_top1["track_ID_imgs"].plot(kind="hist",
                               bins=range(min(df_top1.track_ID_imgs), max(df_top1.track_ID_imgs) + 1, 1),
                               edgecolor="black",
                               title="Distribution of the number of images per tracking ID")
                         .set(xlabel="Number of images per tracking ID"))
plt.savefig(f"{csv_name}_plots/imgs_per_track.png", bbox_inches="tight")
plt.clf()

# Plot histogram with the distribution of the duration [s] per tracking ID
(df_top1["duration_s"].plot(kind="hist",
                            bins=np.arange(min(df_top1.duration_s), max(df_top1.duration_s) + 1, 1),
                            edgecolor="black",
                            title="Distribution of the duration [s] per tracking ID")
                      .set(xlabel="Duration [s] per tracking ID"))
plt.savefig(f"{csv_name}_plots/duration_per_track.png", bbox_inches="tight")
plt.clf()

# Plot barplot with the number of individual tracking IDs per recording
(df_top1.groupby(["date", "rec_ID"])["track_ID"]
        .nunique()
        .plot(kind="bar",
              edgecolor="black",
              ylabel="Number of individual tracking IDs",
              title="Number of individual tracking IDs per recording"))
plt.savefig(f"{csv_name}_plots/tracks_per_rec.png", bbox_inches="tight")
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
plt.savefig(f"{csv_name}_plots/top1_classes_per_rec.png", bbox_inches="tight")
plt.clf()

# Plot barplot with the mean detection confidence per top 1 class
(df_top1.groupby(["top1_class"])["det_conf_mean"]
        .mean()
        .sort_values(ascending=False)
        .plot(kind="bar",
              edgecolor="black",
              ylabel="Mean detection confidence",
              title="Mean detection confidence per top 1 class"))
plt.savefig(f"{csv_name}_plots/top1_classes_mean_det_conf.png", bbox_inches="tight")
plt.clf()

# Plot barplot with the mean classification probability per top 1 class
(df_top1.groupby(["top1_class"])["top1_prob_mean"]
        .mean()
        .sort_values(ascending=False)
        .plot(kind="bar",
              edgecolor="black",
              ylabel="Mean classification probability",
              title="Mean classification probability per top 1 class"))
plt.savefig(f"{csv_name}_plots/top1_classes_mean_prob.png", bbox_inches="tight")
plt.clf()

# Plot barplot with the mean tracking duration [s] per top 1 class
(df_top1.groupby(["top1_class"])["duration_s"]
        .mean()
        .sort_values(ascending=False)
        .plot(kind="bar",
              edgecolor="black",
              ylabel="Mean tracking duration [s]",
              title="Mean tracking duration [s] per top 1 class"))
plt.savefig(f"{csv_name}_plots/top1_classes_mean_duration.png", bbox_inches="tight")
plt.clf()

# Plot barplot with the mean number of images per tracking ID classified as top 1 class
(df_top1.groupby(["top1_class"])["top1_imgs"]
        .mean()
        .sort_values(ascending=False)
        .plot(kind="bar",
              edgecolor="black",
              ylabel="Mean number of images per tracking ID",
              title="Mean number of images per tracking ID classified as top 1 class"))
plt.savefig(f"{csv_name}_plots/top1_classes_mean_imgs.png", bbox_inches="tight")
plt.clf()

# Plot boxplot with mean bbox lengths per top 1 class
(df_top1.plot(kind="box",
              column="bbox_length_mean",
              by="top1_class",
              ylabel="Mean bounding box length",
              title="Mean bounding box length (longer side) per top 1 class"))
plt.savefig(f"{csv_name}_plots/top1_classes_bbox_length.png", bbox_inches="tight")
plt.clf()

# Plot boxplot with mean bbox widths per top 1 class
(df_top1.plot(kind="box",
              column="bbox_width_mean",
              by="top1_class",
              ylabel="Mean bounding box width",
              title="Mean bounding box width (shorter side) per top 1 class"))
plt.savefig(f"{csv_name}_plots/top1_classes_bbox_width.png", bbox_inches="tight")
plt.clf()

# Print info messages to console
path_tables = Path.cwd() / f"{csv_name}_tables"
path_plots = Path.cwd() / f"{csv_name}_plots"
print(f"Analyzed .csv tables saved to: {path_tables}")
print(f"Overview plots saved to:       {path_plots}")
print(f"Script run time: {round(time.monotonic() - start_time, 3)} s")
