# Model training and classification + analysis

<img src="https://raw.githubusercontent.com/maxsitt/insect-detect-docs/main/docs/assets/logo.png" width="500">

[![DOI](https://zenodo.org/badge/580963598.svg)](https://zenodo.org/badge/latestdoi/580963598)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

This repository contains Jupyter notebooks that can be used to train custom
[YOLOv5](https://github.com/ultralytics/yolov5), [YOLOv6](https://github.com/meituan/YOLOv6),
[YOLOv7](https://github.com/WongKinYiu/yolov7) and [YOLOv8](https://github.com/ultralytics/ultralytics)
object detection models or a custom YOLOv5 [image classification](https://github.com/ultralytics/yolov5#classification)
model. All notebooks can be run in [Google Colab](https://colab.research.google.com/),
where you will have access to a free cloud GPU for fast training without special hardware requirements.

You will also find Python scripts for classification ([`predict_mod.py`](https://github.com/maxsitt/insect-detect-ml/blob/main/predict_mod.py))
of insect images (e.g. [cropped detections](https://maxsitt.github.io/insect-detect-docs/deployment/detection/#processing-pipeline))
and analysis ([`csv_analysis.py`](https://github.com/maxsitt/insect-detect-ml/blob/main/csv_analysis.py))
of the [metadata .csv](https://maxsitt.github.io/insect-detect-docs/deployment/detection/#metadata-csv)
files, which are generated as output from the Insect Detect DIY camera trap.

---

## Model training

You can find more information about detection model training
in the [**Insect Detect Docs**](https://maxsitt.github.io/insect-detect-docs/modeltraining/train_detection/) 📑.

- **YOLOv5 detection model training** &nbsp;
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxsitt/insect-detect-ml/blob/main/notebooks/YOLOv5_detection_training_OAK_conversion.ipynb)
- **YOLOv6 detection model training** &nbsp;
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxsitt/insect-detect-ml/blob/main/notebooks/YOLOv6_detection_training.ipynb)
- **YOLOv7 detection model training** &nbsp;
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxsitt/insect-detect-ml/blob/main/notebooks/YOLOv7_detection_training.ipynb)
- **YOLOv8 detection model training** &nbsp;
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxsitt/insect-detect-ml/blob/main/notebooks/YOLOv8_detection_training.ipynb)

The notebooks for detection model training also include all necessary steps to convert your model
to [.blob format](https://docs.luxonis.com/en/latest/pages/model_conversion/) for on-device inference
with the [Luxonis OAK](https://docs.luxonis.com/projects/hardware/en/latest/pages/BW1093.html) cameras.

You can find more information about classification model training
in the [**Insect Detect Docs**](https://maxsitt.github.io/insect-detect-docs/modeltraining/train_classification/) 📑.

- **YOLOv5 classification model training** &nbsp;
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxsitt/insect-detect-ml/blob/main/notebooks/YOLOv5_classification_training.ipynb)

The notebook for classification model training includes [export](https://github.com/ultralytics/yolov5/issues/251)
to [ONNX format](https://onnx.ai/) for faster CPU inference on your local PC.

---

## Classification

The classification script [`predict_mod.py`](https://github.com/maxsitt/insect-detect-ml/blob/main/predict_mod.py)
is a modified version of the [`predict.py`](https://github.com/ultralytics/yolov5/blob/master/classify/predict.py)
script from the [YOLOv5 repo](https://github.com/ultralytics/yolov5/tree/master/classify),
with the following added options:

- `--concat-csv` to concatenate metadata .csv files and append classification results to new columns
- `--new-csv` to create a new .csv file with classification results, e.g. if no metadata .csv files are available
- `--sort-top1` to sort classified images to folders with predicted top1 class as folder name

More information about deployment of the classification script on your PC can be found in the
[**Insect Detect Docs**](https://maxsitt.github.io/insect-detect-docs/deployment/classification/) 📑.

### Classification model

| Model<br><sup> | size<br><sup>(pixels) | Top1 Accuracy<sup>val<br> | Top5 Accuracy<sup>val<br> |
| -------------- | --------------------- | ------------------------- | ------------------------- |
| YOLOv5s-cls    | 128                   | 0.9835                    | 1                         |

**Table Notes**

- The model was trained to 100 epochs with batch size 64 and default settings and hyperparameters.
  Reproduce the model training with the provided
  [Google Colab notebook](https://colab.research.google.com/github/maxsitt/insect-detect-ml/blob/main/notebooks/YOLOv5_classification_training.ipynb).
- Trained on custom [dataset](https://universe.roboflow.com/maximilian-sittinger/insect_detect_classification/dataset/2)
  with 7 classes ([class balance](https://universe.roboflow.com/maximilian-sittinger/insect_detect_classification/health)).

### Validation results on dataset [test split](https://universe.roboflow.com/maximilian-sittinger/insect_detect_classification/browse?queryText=split%3Atest)

| Class       | Number images | Top1 Accuracy<sup>test<br> | Top5 Accuracy<sup>test<br> |
| ----------- | ------------- | -------------------------- | -------------------------- |
| all         | 242           | 0.992                      | 0.996                      |
| episyr_balt | 34            | 1                          | 1                          |
| hovfly      | 12            | 1                          | 1                          |
| fly         | 49            | 0.98                       | 1                          |
| wasp        | 64            | 1                          | 1                          |
| hbee        | 41            | 1                          | 1                          |
| other       | 24            | 0.958                      | 0.958                      |
| shadow      | 18            | 1                          | 1                          |

More information about the respective classes:
[Insect_Detect_classification dataset](https://universe.roboflow.com/maximilian-sittinger/insect_detect_classification)

---

## Analysis

The analysis script [`csv_analysis.py`](https://github.com/maxsitt/insect-detect-ml/blob/main/csv_analysis.py)
can be used to automatically post-process and analyze the concatenated metadata .csv
file after the [classification](https://maxsitt.github.io/insect-detect-docs/deployment/classification/)
step, as it will still contain multiple rows for each tracked insect.
Running the script will yield a .csv file in which each row corresponds to an
individual tracked insect and its classification result with the overall highest
probability. Additionally, several
[plots](https://maxsitt.github.io/insect-detect-docs/deployment/analysis/#overview-plots)
are generated that can give a first overview of the analyzed data.

More information about deployment of the analysis script can be found in the
[**Insect Detect Docs**](https://maxsitt.github.io/insect-detect-docs/deployment/analysis/) 📑.

---

## License

All Python scripts are licensed under the GNU General Public License v3.0
([GNU GPLv3](https://www.gnu.org/licenses/gpl-3.0)).

## Citation

You can cite this repository as:

```
Sittinger, M. (2023). Insect Detect ML - Software for classification of images and analysis
of metadata from a DIY camera trap system (v1.1). Zenodo. https://doi.org/10.5281/zenodo.7603476
```
