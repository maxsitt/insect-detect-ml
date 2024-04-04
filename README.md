# Insect Detect ML - Model training and data processing

<img src="https://raw.githubusercontent.com/maxsitt/insect-detect-docs/main/docs/assets/logo.png" width="540">

[![DOI PLOS ONE](https://img.shields.io/badge/PLOS%20ONE-10.1371%2Fjournal.pone.0295474-BD3094)](https://doi.org/10.1371/journal.pone.0295474)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://choosealicense.com/licenses/agpl-3.0/)
[![DOI Zenodo](https://zenodo.org/badge/580963598.svg)](https://zenodo.org/badge/latestdoi/580963598)

This repository contains Jupyter notebooks that can be used to train custom
[YOLOv5](https://github.com/ultralytics/yolov5), [YOLOv6](https://github.com/meituan/YOLOv6),
[YOLOv7](https://github.com/WongKinYiu/yolov7) and [YOLOv8](https://github.com/ultralytics/ultralytics)
object detection models or a custom [YOLOv5 image classification](https://github.com/ultralytics/yolov5#classification)
model. All notebooks can be run in [Google Colab](https://colab.research.google.com/),
where you will have access to a free cloud GPU for fast training without special hardware requirements.

The Python script for [classification](https://github.com/maxsitt/yolov5/blob/master/classify/predict.py)
of the captured insect images is available in the custom [`yolov5`](https://github.com/maxsitt/yolov5)
fork and can be used together with the provided
[insect classification model](https://github.com/maxsitt/insect-detect-ml/tree/main/models).

Use the [`process_metadata.py`](https://github.com/maxsitt/insect-detect-ml/blob/main/process_metadata.py)
script for post-processing of metadata .csv files with classification results.

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

More information about deployment of the classification script can be found at the
[**Insect Detect Docs**](https://maxsitt.github.io/insect-detect-docs/deployment/classification/) 📑.

&nbsp;

### Classification model

| Model<br><sup>(.onnx) | size<br><sup>(pixels) | Top1 Accuracy<sup>test<br> | Precision<sup>test<br> | Recall<sup>test<br> | F1 score<sup>test<br> |
| --------------------- | --------------------- | -------------------------- | ---------------------- | ------------------- | --------------------- |
| EfficientNet-B0       | 128                   | 0.972                      | 0.971                  | 0.967               | 0.969                 |

**Table Notes**

- The [model](https://github.com/maxsitt/insect-detect-ml/tree/main/models) was trained to 20 epochs with image
  size 128, batch size 64 and default settings and hyperparameters. Reproduce the model training with the provided
  [Google Colab notebook](https://colab.research.google.com/github/maxsitt/insect-detect-ml/blob/main/notebooks/YOLOv5_classification_training.ipynb).
- Trained on [Insect Detect - insect classification dataset v2](https://doi.org/10.5281/zenodo.8325383)
  with 27 classes. To reproduce the dataset split, keep the default settings in the Colab notebook
  (train/val/test ratio = 0.7/0.2/0.1, random seed = 1).
- Dataset can be explored at [Roboflow Universe](https://universe.roboflow.com/maximilian-sittinger/insect_detect_classification_v2).
  Export from Roboflow compresses the images and can lead to a decreased model accuracy.
  It is recommended to use the uncompressed dataset from [Zenodo](https://doi.org/10.5281/zenodo.8325383).

<details>
  <summary>Full model metrics on dataset test split (click to expand)</summary>

| Class        | Images | Top1 Accuracy<sup>test<br> | Precision<sup>test<br> | Recall<sup>test<br> | F1 score<sup>test<br> |
| ------------ | ------ | -------------------------- | ---------------------- | ------------------- | --------------------- |
| all          | 2125   | 0.972                      | 0.971                  | 0.967               | 0.969                 |
| ant          | 111    | 1.0                        | 0.991                  | 1.0                 | 0.996                 |
| bee          | 107    | 0.963                      | 0.972                  | 0.963               | 0.967                 |
| bee_apis     | 31     | 1.0                        | 0.969                  | 1.0                 | 0.984                 |
| bee_bombus   | 127    | 1.0                        | 0.992                  | 1.0                 | 0.996                 |
| beetle       | 52     | 0.885                      | 0.92                   | 0.885               | 0.902                 |
| beetle_cocci | 78     | 0.987                      | 1.0                    | 0.987               | 0.994                 |
| beetle_oedem | 21     | 0.905                      | 0.905                  | 0.905               | 0.905                 |
| bug          | 39     | 0.846                      | 1.0                    | 0.846               | 0.917                 |
| bug_grapho   | 19     | 1.0                        | 1.0                    | 1.0                 | 1.0                   |
| fly          | 173    | 0.971                      | 0.944                  | 0.971               | 0.957                 |
| fly_empi     | 19     | 1.0                        | 1.0                    | 1.0                 | 1.0                   |
| fly_sarco    | 33     | 0.909                      | 0.938                  | 0.909               | 0.923                 |
| fly_small    | 167    | 0.958                      | 0.952                  | 0.958               | 0.955                 |
| hfly_episyr  | 253    | 0.996                      | 0.996                  | 0.996               | 0.996                 |
| hfly_eristal | 197    | 0.99                       | 0.995                  | 0.99                | 0.992                 |
| hfly_eupeo   | 137    | 0.985                      | 0.993                  | 0.985               | 0.989                 |
| hfly_myathr  | 60     | 1.0                        | 1.0                    | 1.0                 | 1.0                   |
| hfly_sphaero | 39     | 0.974                      | 1.0                    | 0.974               | 0.987                 |
| hfly_syrphus | 50     | 0.98                       | 1.0                    | 0.98                | 0.99                  |
| lepi         | 24     | 1.0                        | 0.96                   | 1.0                 | 0.98                  |
| none_bg      | 86     | 0.988                      | 0.966                  | 0.988               | 0.977                 |
| none_bird    | 8      | 1.0                        | 1.0                    | 1.0                 | 1.0                   |
| none_dirt    | 85     | 0.976                      | 0.902                  | 0.976               | 0.938                 |
| none_shadow  | 66     | 0.924                      | 0.953                  | 0.924               | 0.938                 |
| other        | 79     | 0.861                      | 0.883                  | 0.861               | 0.872                 |
| scorpionfly  | 12     | 1.0                        | 1.0                    | 1.0                 | 1.0                   |
| wasp         | 52     | 1.0                        | 1.0                    | 1.0                 | 1.0                   |

<img src="https://raw.githubusercontent.com/maxsitt/insect-detect-docs/main/docs/assets/images/efficientnet-b0_confusion_matrix_test.png" width="800">

</details>

<details>
  <summary>Full model metrics on dataset validation split (click to expand)</summary>

| Class        | Images | Top1 Accuracy<sup>val<br> | Precision<sup>val<br> | Recall<sup>val<br> | F1 score<sup>val<br> |
| ------------ | ------ | ------------------------- | --------------------- | ------------------ | -------------------- |
| all          | 4189   | 0.98                      | 0.979                 | 0.974              | 0.976                |
| ant          | 219    | 0.995                     | 0.995                 | 0.995              | 0.995                |
| bee          | 212    | 0.967                     | 0.958                 | 0.967              | 0.962                |
| bee_apis     | 58     | 1.0                       | 0.967                 | 1.0                | 0.983                |
| bee_bombus   | 252    | 1.0                       | 0.996                 | 1.0                | 0.998                |
| beetle       | 104    | 0.933                     | 0.942                 | 0.933              | 0.937                |
| beetle_cocci | 155    | 1.0                       | 1.0                   | 1.0                | 1.0                  |
| beetle_oedem | 39     | 0.897                     | 0.972                 | 0.897              | 0.933                |
| bug          | 78     | 0.949                     | 0.961                 | 0.949              | 0.955                |
| bug_grapho   | 37     | 1.0                       | 1.0                   | 1.0                | 1.0                  |
| fly          | 343    | 0.983                     | 0.939                 | 0.983              | 0.96                 |
| fly_empi     | 35     | 1.0                       | 0.972                 | 1.0                | 0.986                |
| fly_sarco    | 63     | 0.841                     | 0.964                 | 0.841              | 0.898                |
| fly_small    | 332    | 0.97                      | 0.982                 | 0.97               | 0.976                |
| hfly_episyr  | 503    | 0.996                     | 0.996                 | 0.996              | 0.996                |
| hfly_eristal | 390    | 1.0                       | 1.0                   | 1.0                | 1.0                  |
| hfly_eupeo   | 271    | 0.989                     | 0.993                 | 0.989              | 0.991                |
| hfly_myathr  | 118    | 0.992                     | 1.0                   | 0.992              | 0.996                |
| hfly_sphaero | 74     | 1.0                       | 0.987                 | 1.0                | 0.993                |
| hfly_syrphus | 97     | 1.0                       | 0.99                  | 1.0                | 0.995                |
| lepi         | 45     | 0.978                     | 0.978                 | 0.978              | 0.978                |
| none_bg      | 170    | 0.988                     | 0.982                 | 0.988              | 0.985                |
| none_bird    | 13     | 1.0                       | 1.0                   | 1.0                | 1.0                  |
| none_dirt    | 167    | 0.982                     | 0.976                 | 0.982              | 0.979                |
| none_shadow  | 129    | 0.969                     | 0.984                 | 0.969              | 0.977                |
| other        | 158    | 0.88                      | 0.903                 | 0.88               | 0.891                |
| scorpionfly  | 24     | 1.0                       | 1.0                   | 1.0                | 1.0                  |
| wasp         | 103    | 0.99                      | 1.0                   | 0.99               | 0.995                |

</details>

<img src="https://raw.githubusercontent.com/maxsitt/insect-detect-docs/main/docs/assets/images/classification_classes.png" width="800">

---

## Metadata post-processing

Install the required packages by running:

```
python.exe -m pip install -r requirements.txt
```

Or use the Python Launcher for Windows with:

```
py -m pip install -r requirements.txt
```

The [`process_metadata.py`](https://github.com/maxsitt/insect-detect-ml/blob/main/process_metadata.py)
script can be used to automatically post-process the concatenated metadata .csv file after the
[classification](https://maxsitt.github.io/insect-detect-docs/deployment/classification/)
step, as it will still contain multiple rows for each tracked insect.

The output of the script includes a `*top1_final.csv` file in which each row
corresponds to an individual tracked insect and its classification result
with the highest weighted probability. Additionally, several
[plots](https://maxsitt.github.io/insect-detect-docs/deployment/post-processing/#overview-plots)
are generated that can give a first overview of the processed metadata.

More information about deployment of the post-processing script can be found at the
[**Insect Detect Docs**](https://maxsitt.github.io/insect-detect-docs/deployment/post-processing/) 📑.

---

## Image processing

The [`process_images.py`](https://github.com/maxsitt/insect-detect-ml/blob/main/process_images.py)
script can be used to calculate different metrics of the captured images
(e.g. mean/median/min/max width/height) and remove corrupted .jpg images
from the data folder (camera trap output). These can be rarely generated
by the automated monitoring script (e.g. in power outage situations) and
will cause an error while running the classification script.

---

## License

This repository is licensed under the terms of the GNU Affero General Public
License v3.0 ([GNU AGPLv3](https://choosealicense.com/licenses/agpl-3.0/)).

## Citation

If you use resources from this repository, please cite our paper:

```
Sittinger M, Uhler J, Pink M, Herz A (2024) Insect detect: An open-source DIY camera trap for automated insect monitoring. PLoS ONE 19(4): e0295474. https://doi.org/10.1371/journal.pone.0295474
```
