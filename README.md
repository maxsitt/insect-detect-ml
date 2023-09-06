# Insect Detect ML - Model training and data processing

<img src="https://raw.githubusercontent.com/maxsitt/insect-detect-docs/main/docs/assets/logo.png" width="500">

[![DOI](https://zenodo.org/badge/580963598.svg)](https://zenodo.org/badge/latestdoi/580963598)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://choosealicense.com/licenses/agpl-3.0/)

This repository contains Jupyter notebooks that can be used to train custom
[YOLOv5](https://github.com/ultralytics/yolov5), [YOLOv6](https://github.com/meituan/YOLOv6),
[YOLOv7](https://github.com/WongKinYiu/yolov7) and [YOLOv8](https://github.com/ultralytics/ultralytics)
object detection models or a custom [YOLOv5 image classification](https://github.com/ultralytics/yolov5#classification)
model. All notebooks can be run in [Google Colab](https://colab.research.google.com/),
where you will have access to a free cloud GPU for fast training without special hardware requirements.

The Python script for [classification](https://github.com/maxsitt/yolov5/blob/master/classify/predict.py)
of the captured insect images is available in the custom [`yolov5`](https://github.com/maxsitt/yolov5)
fork and can be used together with the provided `yolov5s-cls_128.onnx` insect classification model.

Use the [`csv_analysis.py`](https://github.com/maxsitt/insect-detect-ml/blob/main/csv_analysis.py)
script for post-processing and analysis of the metadata .csv file with classification results.

---

## Model training

You can find more information about detection model training
at the [**Insect Detect Docs**](https://maxsitt.github.io/insect-detect-docs/modeltraining/train_detection/) 📑.

- **YOLOv5 detection model training** &nbsp;
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxsitt/insect-detect-ml/blob/main/notebooks/YOLOv5_detection_training.ipynb)
- **YOLOv6 detection model training** &nbsp;
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxsitt/insect-detect-ml/blob/main/notebooks/YOLOv6_detection_training.ipynb)
- **YOLOv7 detection model training** &nbsp;
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxsitt/insect-detect-ml/blob/main/notebooks/YOLOv7_detection_training.ipynb)
- **YOLOv8 detection model training** &nbsp;
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxsitt/insect-detect-ml/blob/main/notebooks/YOLOv8_detection_training.ipynb)

  > The PyTorch model weights can be converted to [.blob](https://docs.luxonis.com/en/latest/pages/model_conversion/)
    format at [tools.luxonis.com](https://tools.luxonis.com/) for on-device inference
    with the [Luxonis OAK](https://docs.luxonis.com/projects/hardware/en/latest/) devices.

&nbsp;

You can find more information about classification model training
at the [**Insect Detect Docs**](https://maxsitt.github.io/insect-detect-docs/modeltraining/train_classification/) 📑.

- **YOLOv5 classification model training** &nbsp;
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxsitt/insect-detect-ml/blob/main/notebooks/YOLOv5_classification_training.ipynb)

  > The notebook for classification model training includes [export](https://github.com/ultralytics/yolov5/issues/251)
    to [ONNX](https://onnx.ai/) format for faster CPU inference.

---

## Classification

The modified [classification script](https://github.com/maxsitt/yolov5/blob/master/classify/predict.py)
in the custom [`yolov5`](https://github.com/maxsitt/yolov5) fork includes the following added options:

- `--sort-top1` sort the classified images to folders with the predicted top1 class as folder name
- `--sort-prob` sort images first by probability and then by top1 class (requires --sort-top1)
- `--concat-csv` concatenate all metadata .csv files and append classification results to new columns
- `--new-csv` create a new .csv file with classification results, e.g. if no metadata .csv files are available

More information about deployment of the classification script can be found at the
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

## Post-processing & analysis

Install the required packages by running:

```
python.exe -m pip install -r requirements.txt
```

The [`csv_analysis.py`](https://github.com/maxsitt/insect-detect-ml/blob/main/csv_analysis.py)
script can be used to automatically post-process and analyze the concatenated metadata .csv
file after the [classification](https://maxsitt.github.io/insect-detect-docs/deployment/classification/)
step, as it will still contain multiple rows for each tracked insect.

The output of the script includes a final .csv file in which each row corresponds to
an individual tracked insect and its classification result with the overall highest
probability. Additionally, several
[plots](https://maxsitt.github.io/insect-detect-docs/deployment/analysis/#overview-plots)
are generated that can give a first overview of the analyzed data.

More information about deployment of the analysis script can be found at the
[**Insect Detect Docs**](https://maxsitt.github.io/insect-detect-docs/deployment/analysis/) 📑.

---

## Check images

The [`check_images.py`](https://github.com/maxsitt/insect-detect-ml/blob/main/check_images.py)
script can be used to calculate different metrics of the captured images
(e.g. mean/median/min/max width/height) and remove corrupted .jpg images
from the data folder (camera trap output). These are rarely generated by
the automated monitoring script and will cause an error while running the
classification script.

---

## License

All Python scripts are licensed under the GNU Affero General Public License v3.0
([GNU AGPLv3](https://choosealicense.com/licenses/agpl-3.0/)).

## Citation

You can cite this repository as:

```
Sittinger, M. (2023). Insect Detect ML - Software for classification of images and analysis
of metadata from a DIY camera trap system (v1.2). Zenodo. https://doi.org/10.5281/zenodo.7502195
```
