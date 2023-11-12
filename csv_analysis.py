'''
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Website:  https://maxsitt.github.io/insect-detect-docs/
License:  GNU AGPLv3 (https://choosealicense.com/licenses/agpl-3.0/)

This Python script does the following:
- read "*metadata_classified.csv" file into pandas DataFrame
- calculate relative (or absolute, if frame width/height in mm is given) bounding box sizes
  and bbox length (= longer side) + bbox width (= shorter side)
- save metadata sorted by recording ID, tracking ID and timestamp successively (ascending)
  to "*metadata_classified_sorted.csv"

- group by tracking ID per recording ID and calculate per tracking ID (per recording ID):
  - number of images per top1 class
  - total number of images per tracking ID
  - mean classification probability for each top1 class
  - weighted mean classification probability for each top1 class
    (number images top1 class / total number images tracking ID) * mean classification probability
  - mean bbox length and bbox width for each top1 class
- save metadata sorted by recording ID, tracking ID (ascending) and weighted
  probability (descending) to "*metadata_classified_top1_all.csv"

- group by tracking ID per recording ID and calculate/extract per tracking ID (per recording ID):
  - total number of images
  - date from first timestamp
  - first and last timestamp (start/end time)
  - duration [s] (end time - start time)
  - mean detection confidence
  - number of images of the top1 class with the highest weighted probability
  - name of the top1 class with the highest weighted probability
  - mean classification probability of the top1 class with the highest weighted probability
  - weighted classification probability of the top1 class with the highest weighted probability
  - mean bounding box length and width of the top1 class with the highest weighted probability
- remove tracking IDs with less or more than the specified number of images
- save metadata calculated for each tracking ID (per recording ID) (one row per tracking ID)
  to "*metadata_classified_top1_final.csv"

- extract some info about the analysis and save to "*metadata_classified_analysis_info.csv"
  - total number of unique tracking IDs per recording ID
  - total number of unique top1 classes per recording ID
  - number of unique trackings IDs per recording ID left after removing all
    tracking IDs with less or more than the specified number of images
  - number of unique top1 classes per recording ID left after removing all
    tracking IDs with less or more than the specified number of images
  - minimum number of images per tracking ID to keep it in final .csv
  - maximum number of images per tracking ID to keep it in final .csv

- create and save some basic plots for a quick first data overview

- optional arguments:
  "-csv [path]" path to folder containing metadata .csv file(s), will
                return the last metadata .csv file in alphabetical order
  "-width [mm]" absolute frame width in mm to calculate "true"
                bbox size [mm] (default of 1 gives relative bbox size)
  "-height [mm]" absolute frame height in mm to calculate "true"
                 bbox size [mm] (default of 1 gives relative bbox size)
  "-min_tracks [number]" remove tracking IDs with less than the
                         specified number of images (default = 3)
  "-max_tracks [number]" remove tracking IDs with more than the
                         specified number of images (default = 1800)
'''

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import distinctipy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set start time for script execution timer
start_time = time.monotonic()

# Define optional arguments
parser = argparse.ArgumentParser()
parser.add_argument("-csv", "--csv_path", type=str,
    help="path to folder containing metadata .csv file(s)")
parser.add_argument("-width", "--frame_width", type=int, default=1,
    help="absolute frame width [mm] (default of 1 gives relative bbox size)")
parser.add_argument("-height", "--frame_height", type=int, default=1,
    help="absolute frame height [mm] (default of 1 gives relative bbox size)")
parser.add_argument("-min_tracks", "--remove_min_tracks", type=int, default=3,
    help="remove tracking IDs with less than the specified number of images")
parser.add_argument("-max_tracks", "--remove_max_tracks", type=int, default=1800,
    help="remove tracking IDs with more than the specified number of images")
args = parser.parse_args()

# Get metadata .csv file with classification results
if args.csv_path is not None:
    csv_path = Path(args.csv_path)
else:
    csv_path = Path.cwd()

csv_files = list(csv_path.glob("*metadata_classified.csv"))

if len(csv_files) > 0:
    print(f"\nFound {len(csv_files)} metadata .csv file(s) in {csv_path}\n")
    csv_file = csv_files[-1]
    csv_name = csv_file.stem
else:
    print(f"\nCould not find any metadata .csv file(s) in {csv_path}\n")
    sys.exit()

# Create folder to save post-processed .csv files and plots
timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S")
save_path = f"{csv_file.parent}/{timestamp}_metadata_analyzed"
Path(save_path).mkdir(parents=True, exist_ok=True)

# Read metadata .csv into pandas DataFrame and convert timestamp to pandas datetime object
df = pd.read_csv(csv_file, encoding="utf-8")
df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y%m%d_%H-%M-%S.%f")

# Calculate bounding box sizes (default of 1 gives relative bbox size)
df["bbox_size_x"] = round((df["x_max"] - df["x_min"]) * args.frame_width, 4)
df["bbox_size_y"] = round((df["y_max"] - df["y_min"]) * args.frame_height, 4)

# Get bbox length (= longer side) and bbox width (= shorter side)
df["bbox_length"] = np.where(df["bbox_size_x"] > df["bbox_size_y"], df["bbox_size_x"], df["bbox_size_y"])
df["bbox_width"] = np.where(df["bbox_size_x"] < df["bbox_size_y"], df["bbox_size_x"], df["bbox_size_y"])

# Sort metadata by recording ID, tracking ID and timestamp successively (ascending) and write to .csv
df = df.sort_values(["rec_ID", "track_ID", "timestamp"])
df.to_csv(f"{save_path}/{csv_name}_sorted.csv", index=False)


# Create DataFrame with recording ID, tracking ID and top1 classes + number of images per top1 class
df_top1_all = (df.groupby(["rec_ID", "track_ID"])["top1"]
                 .value_counts(sort=False)
                 .reset_index(name="top1_imgs"))

# Sum the number of images of all top1 classes per tracking ID
df_top1_all["track_ID_imgs"] = (df_top1_all.groupby(["rec_ID", "track_ID"])["top1_imgs"]
                                           .transform("sum")
                                           .reset_index(drop=True))

# Calculate mean classification probability for each top1 class per tracking ID
df_top1_all["top1_prob_mean"] = (df.groupby(["rec_ID", "track_ID", "top1"])["top1_prob"]
                                   .mean()
                                   .round(2)
                                   .reset_index(drop=True))

# Calculate weighted mean classification probability for each top1 class per tracking ID
df_top1_all["top1_prob_weighted"] = ((df_top1_all["top1_imgs"] / df_top1_all["track_ID_imgs"]) * df_top1_all["top1_prob_mean"]).round(2)

# Calculate mean bounding box length and width for each top1 class per tracking ID
df_top1_all["bbox_length_mean"] = (df.groupby(["rec_ID", "track_ID", "top1"])["bbox_length"]
                                     .mean()
                                     .round(3)
                                     .reset_index(drop=True))
df_top1_all["bbox_width_mean"] = (df.groupby(["rec_ID", "track_ID", "top1"])["bbox_width"]
                                    .mean()
                                    .round(3)
                                    .reset_index(drop=True))

# Sort data by recording ID, tracking ID (ascending) and weighted probability (descending) successively and write to .csv
df_top1_all = df_top1_all.sort_values(["rec_ID", "track_ID", "top1_prob_weighted"], ascending=[True, True, False])
df_top1_all = df_top1_all[["rec_ID", "track_ID", "track_ID_imgs", "top1_imgs", "top1",
                           "top1_prob_mean", "top1_prob_weighted", "bbox_length_mean", "bbox_width_mean"]]
df_top1_all.to_csv(f"{save_path}/{csv_name}_top1_all.csv", index=False)


# Create DataFrame with recording ID and tracking ID + number of images per tracking ID
df_top1 = (df.groupby(["rec_ID", "track_ID"])["track_ID"]
             .count()
             .reset_index(name="track_ID_imgs"))

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

# Calculate duration of each tracking ID in seconds
df_top1["duration_s"] = ((df_top1["end_time"] - df_top1["start_time"]) / pd.Timedelta(seconds=1)).round(2)

# Calculate mean detection confidence of each tracking ID
df_top1["det_conf_mean"] = (df.groupby(["rec_ID", "track_ID"])["confidence"]
                              .mean()
                              .round(2)
                              .reset_index(drop=True))

# Get number of images, name, mean + weighted classification probability of the top1 class with the highest weighted probability
df_top1["top1_imgs"] = (df_top1_all.loc[df_top1_all
                                   .groupby(["rec_ID", "track_ID"])["top1_prob_weighted"]
                                   .idxmax()]["top1_imgs"]
                                   .reset_index(drop=True))
df_top1["top1"] = (df_top1_all.loc[df_top1_all
                              .groupby(["rec_ID", "track_ID"])["top1_prob_weighted"]
                              .idxmax()]["top1"]
                              .reset_index(drop=True))
df_top1["top1_prob_mean"] = (df_top1_all.loc[df_top1_all
                                        .groupby(["rec_ID", "track_ID"])["top1_prob_weighted"]
                                        .idxmax()]["top1_prob_mean"]
                                        .reset_index(drop=True))
df_top1["top1_prob_weighted"] = (df_top1_all.groupby(["rec_ID", "track_ID"])["top1_prob_weighted"]
                                            .max()
                                            .reset_index(drop=True))

# Get mean bounding box length and width of the top1 class with the highest weighted probability
df_top1["bbox_length_mean"] = (df_top1_all.loc[df_top1_all
                                          .groupby(["rec_ID", "track_ID"])["top1_prob_weighted"]
                                          .idxmax()]["bbox_length_mean"]
                                          .reset_index(drop=True))
df_top1["bbox_width_mean"] = (df_top1_all.loc[df_top1_all
                                         .groupby(["rec_ID", "track_ID"])["top1_prob_weighted"]
                                         .idxmax()]["bbox_width_mean"]
                                         .reset_index(drop=True))

# Remove tracking IDs with less (default: 3) or more (default: 1800) than the specified number of images
removed_min_tracks = len(df_top1[df_top1["track_ID_imgs"] < args.remove_min_tracks])
removed_max_tracks = len(df_top1[df_top1["track_ID_imgs"] > args.remove_max_tracks])
print(f"Removed {removed_min_tracks} tracking IDs with less than {args.remove_min_tracks} images.")
print(f"Removed {removed_max_tracks} tracking IDs with more than {args.remove_max_tracks} images.\n")
df_top1 = df_top1[(df_top1["track_ID_imgs"] >= args.remove_min_tracks) &
                  (df_top1["track_ID_imgs"] <= args.remove_max_tracks)]

# Sort columns and write to .csv file
df_top1 = df_top1[["rec_ID", "date", "track_ID", "start_time", "end_time", "duration_s",
                   "det_conf_mean", "track_ID_imgs", "top1_imgs", "top1", "top1_prob_mean",
                   "top1_prob_weighted", "bbox_length_mean", "bbox_width_mean"]]
df_top1.to_csv(f"{save_path}/{csv_name}_top1_final.csv", index=False)


# Create DataFrame with info about the analysis and write to .csv
df_info = (df.groupby("rec_ID", as_index=True)
             .agg(track_IDs_total=("track_ID", "nunique"),
                  top1_classes_total=("top1", "nunique")))
df_info["track_IDs_left"] = (df_top1.groupby("rec_ID")
                                    .size())
df_info["top1_classes_left"] = (df_top1.groupby("rec_ID")["top1"]
                                       .nunique())
df_info[["track_IDs_left", "top1_classes_left"]] = (df_info[["track_IDs_left", "top1_classes_left"]].fillna(0)
                                                                                                    .astype(int))
df_info["min_track_imgs"] = args.remove_min_tracks
df_info["max_track_imgs"] = args.remove_max_tracks
df_info.to_csv(f"{save_path}/{csv_name}_analysis_info.csv")


#### Plots ####
print("Creating plots...")

# Draw grid lines behind other elements
plt.rcParams["axes.axisbelow"] = True

# Plot histogram with the distribution of the recording duration per tracking ID
(df_top1["duration_s"].plot(kind="hist",
                            figsize=(15, 10),
                            bins=np.arange(min(df_top1["duration_s"]), max(df_top1["duration_s"]) + 1, 1),
                            title="Distribution of the recording duration per tracking ID")
                      .set(xlabel="Recording duration per tracking ID [s]"))
plt.grid(axis="y", color="gray", linewidth=0.5, alpha=0.2)
plt.savefig(f"{save_path}/duration_per_track.png", dpi=300, bbox_inches="tight")
plt.close()

# Plot histogram with the distribution of the number of images per tracking ID
(df_top1["track_ID_imgs"].plot(kind="hist",
                               figsize=(15, 10),
                               bins=range(min(df_top1["track_ID_imgs"]), max(df_top1["track_ID_imgs"]) + 1, 1),
                               title="Distribution of the number of images per tracking ID")
                         .set(xlabel="Number of images per tracking ID"))
plt.grid(axis="y", color="gray", linewidth=0.5, alpha=0.2)
plt.savefig(f"{save_path}/imgs_per_track.png", dpi=300, bbox_inches="tight")
plt.close()

# Plot boxplot with mean bbox lengths per top 1 class
if args.frame_width != 1 or args.frame_height != 1:
    Y_LABEL_L = "Mean bounding box length [mm]"
else:
    Y_LABEL_L = "Mean relative bounding box length"
(df_top1.plot(kind="box",
              column="bbox_length_mean",
              by="top1",
              rot=90,
              figsize=(15, 10),
              xlabel="Top 1 class",
              ylabel=Y_LABEL_L))
plt.grid(axis="y", color="gray", linewidth=0.5, alpha=0.2)
plt.suptitle("")
plt.title("Mean bounding box length (longer side) per top 1 class")
plt.savefig(f"{save_path}/top1_bbox_length.png", dpi=300, bbox_inches="tight")
plt.close()

# Plot boxplot with mean bbox widths per top 1 class
if args.frame_width != 1 or args.frame_height != 1:
    Y_LABEL_W = "Mean bounding box width [mm]"
else:
    Y_LABEL_W = "Mean relative bounding box width"
(df_top1.plot(kind="box",
              column="bbox_width_mean",
              by="top1",
              rot=90,
              figsize=(15, 10),
              xlabel="Top 1 class",
              ylabel=Y_LABEL_W))
plt.grid(axis="y", color="gray", linewidth=0.5, alpha=0.2)
plt.suptitle("")
plt.title("Mean bounding box width (shorter side) per top 1 class")
plt.savefig(f"{save_path}/top1_bbox_width.png", dpi=300, bbox_inches="tight")
plt.close()

# Plot barplot with the mean detection confidence per top 1 class
(df_top1.groupby(["top1"])["det_conf_mean"]
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
plt.savefig(f"{save_path}/top1_mean_det_conf.png", dpi=300, bbox_inches="tight")
plt.close()

# Plot barplot with the mean tracking duration per top 1 class
(df_top1.groupby(["top1"])["duration_s"]
        .mean()
        .sort_values(ascending=False)
        .plot(kind="bar",
              figsize=(15, 10),
              edgecolor="black",
              xlabel="Top 1 class",
              ylabel="Mean tracking duration [s]",
              title="Mean tracking duration per top 1 class"))
plt.grid(axis="y", color="gray", linewidth=0.5, alpha=0.2)
plt.savefig(f"{save_path}/top1_mean_duration.png", dpi=300, bbox_inches="tight")
plt.close()

# Plot barplot with the mean number of images per tracking ID classified as top 1 class
(df_top1.groupby(["top1"])["top1_imgs"]
        .mean()
        .sort_values(ascending=False)
        .plot(kind="bar",
              figsize=(15, 10),
              edgecolor="black",
              xlabel="Top 1 class",
              ylabel="Mean number of images per tracking ID",
              title="Mean number of images per tracking ID classified as top 1 class"))
plt.grid(axis="y", color="gray", linewidth=0.5, alpha=0.2)
plt.savefig(f"{save_path}/top1_mean_imgs.png", dpi=300, bbox_inches="tight")
plt.close()

# Plot boxplot with the classification probability per top 1 class
(df_top1.plot(kind="box",
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
plt.savefig(f"{save_path}/top1_prob.png", dpi=300, bbox_inches="tight")
plt.close()

# Plot barplot with the mean classification probability per top 1 class
(df_top1.groupby(["top1"])["top1_prob_mean"]
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
plt.savefig(f"{save_path}/top1_prob_mean.png", dpi=300, bbox_inches="tight")
plt.close()

# Plot barplot with the number of individual tracking IDs per recording
(df_top1.groupby(["date", "rec_ID"])["track_ID"]
        .nunique()
        .plot(kind="bar",
              figsize=(15, 10),
              edgecolor="black",
              xlabel="Recording interval [Date, Rec ID]",
              ylabel="Number of individual tracking IDs",
              title="Number of individual tracking IDs per recording"))
plt.grid(axis="y", color="gray", linewidth=0.5, alpha=0.2)
plt.xticks(fontsize="xx-small")
plt.savefig(f"{save_path}/tracks_per_rec.png", dpi=300, bbox_inches="tight")
plt.close()

# Create custom colormap with distinctipy
num_top1 = df_top1["top1"].nunique()
colors = distinctipy.get_colors(num_top1)
cmap_custom = distinctipy.get_colormap(colors)

# Plot stacked barplot with the top 1 classes per recording
(df_top1.groupby(["date", "rec_ID"])["top1"]
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
plt.savefig(f"{save_path}/top1_per_rec.png", dpi=300, bbox_inches="tight")
plt.close()

# Print save path and script run time
print(f"\nPost-processed metadata .csv files and plots saved to: {save_path}")
print(f"\nScript run time: {round(time.monotonic() - start_time, 3)} s\n")
