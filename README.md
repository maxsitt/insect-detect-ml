# Model training and classification + analysis

<img src="https://raw.githubusercontent.com/maxsitt/insect-detect-docs/main/docs/assets/logo.png" width="500">

[![License badge](https://img.shields.io/badge/license-GPLv3-yellowgreen)](https://choosealicense.com/licenses/gpl-3.0/)

This repository contains Jupyter notebooks that can be used to train custom
[YOLOv5](https://github.com/ultralytics/yolov5) detection and classification
models with your own data (annotated images). You will also find Python scripts
for classification (`predict_mod.py`) of insect images and analysis (`csv_analysis.py`) 
of the metadata.csv output from the Insect Detect DIY camera trap.

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

More information about deployment of the classification script `predict_mod.py` can be found in the
[**Insect Detect Docs**](https://maxsitt.github.io/insect-detect-docs/deployment/classification/) 📑.

The classification script is a slightly modified version of the `predict.py` script from the YOLOv5 repo.

## Analysis

More information about deployment of the analysis script `csv_analysis.py` can be found in the
[**Insect Detect Docs**](https://maxsitt.github.io/insect-detect-docs/deployment/classification/) 📑.

## License

All Python scripts are licensed under the GNU General Public License v3.0
([GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)).
