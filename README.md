# Model training and classification + analysis

<img src="https://raw.githubusercontent.com/maxsitt/insect-detect-docs/main/docs/assets/logo.png" width="500">

[![DOI](https://zenodo.org/badge/580963598.svg)](https://zenodo.org/badge/latestdoi/580963598)
[![License badge](https://img.shields.io/badge/license-GPLv3-yellowgreen)](https://choosealicense.com/licenses/gpl-3.0/)

This repository contains Jupyter notebooks that can be used to train custom
[YOLOv5](https://github.com/ultralytics/yolov5) detection and
[classification](https://github.com/ultralytics/yolov5/pull/8956)
models with your own data (annotated images). You will also find Python
scripts for classification ([`predict_mod.py`](https://github.com/maxsitt/insect-detect-ml/blob/main/predict_mod.py))
of insect images
(e.g. [cropped detections](https://maxsitt.github.io/insect-detect-docs/deployment/detection/#processing-pipeline))
and analysis ([`csv_analysis.py`](https://github.com/maxsitt/insect-detect-ml/blob/main/csv_analysis.py)) of the
[metadata .csv](https://maxsitt.github.io/insect-detect-docs/deployment/detection/#metadata-csv)
files, which are generated as output from the Insect Detect DIY camera trap.

## Model training

More information about model training can be found in the
[**Insect Detect Docs**](https://maxsitt.github.io/insect-detect-docs/modeltraining/yolov5/) 📑.

- **YOLOv5 detection model training** &nbsp; &nbsp; &nbsp;
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxsitt/insect-detect-ml/blob/main/notebooks/YOLOv5_detection_training_OAK_conversion.ipynb)
- **YOLOv5 classification model training** &nbsp;
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxsitt/insect-detect-ml/blob/main/notebooks/YOLOv5_classification_training.ipynb)

The notebook for detection model training also includes all necessary steps to convert
your model into the [.blob format](https://docs.luxonis.com/en/latest/pages/model_conversion/)
for on-device inference with the [Luxonis OAK](https://docs.luxonis.com/projects/hardware/en/latest/pages/BW1093.html)
cameras. In the notebook for classification model training, you can
[export](https://github.com/ultralytics/yolov5/issues/251) your
trained model to [ONNX format](https://onnx.ai/) for faster CPU inference on your
local PC. All steps in both notebooks are easy to follow and include interactive forms
that enable you to change the important parameters without having to write any code.

## Classification

The classification script [`predict_mod.py`](https://github.com/maxsitt/insect-detect-ml/blob/main/predict_mod.py)
is a modified version of the [`predict.py`](https://github.com/ultralytics/yolov5/blob/master/classify/predict.py)
script from the [YOLOv5 repo](https://github.com/ultralytics/yolov5/tree/master/classify),
with the following added options:

- `--concat-csv` to concatenate metadata .csv files and append classification results to new columns
- `--new-csv` to create a new .csv file with classification results, e.g. if no metadata .csv files are available
- `--sort-top1` to sort classified images to folders with predicted top1 class as folder name and do not write results on to image as text (which is the default configuration)

More information about deployment of the classification script can be found in the
[**Insect Detect Docs**](https://maxsitt.github.io/insect-detect-docs/deployment/classification/) 📑.

## Classification model

| Model<br><sup>(.pt) | size<br><sup>(pixels) | Top1 Accuracy<sup>val<br> | Top5 Accuracy<sup>val<br>  |
| ------------------- | --------------------- | ------------------------- | -------------------------- |
| **YOLOv5s-cls**     | 128                   | 0.9835                    | 1                          |

**Table Notes**

- The model was trained to 100 epochs with batch size 64 and default settings and hyperparameters.
  Reproduce the model training with the provided
  [Google Colab notebook](https://colab.research.google.com/github/maxsitt/insect-detect-ml/blob/main/notebooks/YOLOv5_classification_training.ipynb).
- Trained on custom [dataset](https://universe.roboflow.com/maximilian-sittinger/insect_detect_classification/dataset/2)
  with 7 classes ([class balance](https://universe.roboflow.com/maximilian-sittinger/insect_detect_classification/health)).

### Validation results on dataset test split

| Class       | Number images | Top1 Accuracy<sup>test<br> | Top5 Accuracy<sup>test<br>  |
| ----------- | ------------- | -------------------------- | --------------------------- |
| all         | 242           | 0.992                      | 0.996                       |
| episyr_balt | 34            | 1                          | 1                           |
| hovfly      | 12            | 1                          | 1                           |
| fly         | 49            | 0.98                       | 1                           |
| wasp        | 64            | 1                          | 1                           |
| hbee        | 41            | 1                          | 1                           |
| other       | 24            | 0.958                      | 0.958                       |
| shadow      | 18            | 1                          | 1                           |

More information about the respective classes:
[Insect_Detect_classification dataset](https://universe.roboflow.com/maximilian-sittinger/insect_detect_classification)

Take a look at the dataset test split:
[Insect_Detect_classification dataset test split](https://universe.roboflow.com/maximilian-sittinger/insect_detect_classification/browse?queryText=split%3Atest)

## Analysis

The analysis script [`csv_analysis.py`](https://github.com/maxsitt/insect-detect-ml/blob/main/csv_analysis.py)
can be used to automatically analyze the concatenated metadata .csv file after the
classification step, as it will still contain multiple rows for each tracked insect.
Running the script will yield a .csv file in which each row corresponds to an
individual tracked insect and its classification result with the overall highest
probability. Additionally, several
[plots](https://maxsitt.github.io/insect-detect-docs/deployment/analysis/#overview-plots)
are generated that can give a first overview of the analyzed data.

More information about deployment of the analysis script can be found in the
[**Insect Detect Docs**](https://maxsitt.github.io/insect-detect-docs/deployment/analysis/) 📑.

## License

All Python scripts are licensed under the GNU General Public License v3.0
([GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)).

## Citation

You can cite this repository as:

```
Sittinger, M. (2023). Insect Detect ML - Software for classification of images and analysis
of metadata from a DIY camera trap system (v1.1). Zenodo. https://doi.org/10.5281/zenodo.7603476
```
