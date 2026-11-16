# PA04: Effect of Training Data Size and Class Imbalance on Plastic-vs-Metal Waste Classification

Course: EF2039 Introduction to AI Programming (KENTECH) - PA04
Author: **Kyuho Shim (20250946)**
Repository: <https://github.com/Kiren-del/EF2039_AI_Programming_Kyuho-Shim>

This repository contains the models, test data, evaluation code and results for a small
data-centric study with a binary image classifier (plastic vs. metal waste), trained with
[Teachable Machine](https://teachablemachine.withgoogle.com/) and evaluated on a fixed held-out test set.

## Research questions

- **RQ1 (data size):** How does the number of training images per class (10, 30, 100, 300) affect accuracy?
- **RQ2 (class imbalance):** How does a shrinking minority class (metal: 100, 20, 5 images; plastic fixed at 100) affect
  minority-class recall and overall accuracy?

## Repository structure

```
.
├── README.md
├── environment.yml              # Conda environment
├── evaluate_models.py           # evaluates all models on the fixed test set
├── dataset.zip                  # fixed test set (50 plastic + 50 metal images); unzip before running -> dataset/
├── test_split.csv               # exact test file list
├── training_data.md             # training-set composition and caveats
├── model/<NAME>/converted_keras/{keras_model.h5, labels.txt}   # trained weights (M1-M5, B2, B3)
├── results/
│   ├── summary.csv              # all metrics per model
│   ├── predictions_<NAME>.csv   # per-image predictions (written when you run the script)
│   └── *.png                    # confusion matrices, RQ1 / RQ2 plots
```

After unzipping `dataset.zip`, the folder `dataset/` contains `plastic/` (50 images) and `metal/` (50 images).

## Data

- **Source:** Kaggle "Garbage classification" dataset
  ([Garbage Classification](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)). Only the `plastic` and `metal` classes are used.
  Please refer to the Kaggle page for the dataset's license and terms of use.
- **Image size:** 512 x 384 (JPEG). Teachable Machine resizes inputs to 224 x 224 internally.
- **Class counts in the source data:** plastic 482, metal 410.
- **Fixed test set (all models):** 50 plastic (`plastic1-50`) + 50 metal (`metal1-50`) images, listed in `test_split.csv`.
  The 50 metal test images were removed from the pool used for training (training metal images come from `metal51-410`).
- **Training sets:** drawn at random from the remaining images (plastic: `plastic51-482`, metal: `metal51-410`), so no test image is used for training
  (except M3, see below). Each training set was sampled independently, i.e. the smaller sets are not subsets of the larger ones.
  See [`training_data.md`](training_data.md).

## Models and training settings

All models are Teachable Machine image classifiers (transfer learning on a pretrained backbone) exported as Keras models.
Settings are identical for every model (Teachable Machine defaults):

| Setting | Value |
|---|---|
| Epochs | 50 |
| Batch size | 16 |
| Learning rate | 0.001 |
| Classes | plastic, metal |
| Validation split | Teachable Machine's built-in split (not used for the reported results) |

| Model | Purpose | Plastic / metal training images |
|---|---|---|
| M1 | RQ1 | 10 / 10 |
| M2 | RQ1 | 30 / 30 |
| M3 | - (excluded) | 50 / 50, trained on the same images as the test set; its score is not reported |
| M4 (= B1) | RQ1, RQ2 baseline | 100 / 100 |
| M5 | RQ1 | 300 / 300 |
| B2 | RQ2 | 100 / 20 |
| B3 | RQ2 | 100 / 5 |

## Environment

```bash
git clone https://github.com/Kiren-del/EF2039_AI_Programming_Kyuho-Shim
cd EF2039_AI_Programming_Kyuho-Shim
conda env create -f environment.yml
conda activate pa04-garbage
```

Hardware: CPU only is sufficient (inference on 100 images x 7 models).

Tested environment (Windows): Python 3.10, TensorFlow 2.15.0, Keras 2.15.0, NumPy 1.26.4, Pillow 12.3.0, Matplotlib 3.10.9.

## Reproduce the results

1. Unzip the test set so that the folders `dataset/plastic` and `dataset/metal` exist next to `evaluate_models.py`
   (Windows PowerShell: `Expand-Archive dataset.zip -DestinationPath .`, Linux/macOS: `unzip dataset.zip`).
2. Run the evaluation:

```bash
python evaluate_models.py
```

This loads every model in `model/`, runs it on the 100 test images, and writes `results/summary.csv`,
`results/predictions_<NAME>.csv`, `results/confusion_matrices.png`, `results/rq1_data_size.png` and `results/rq2_imbalance.png`.
Preprocessing matches Teachable Machine's export: center-crop/resize to 224 x 224, then `(x / 127.5) - 1`.

Retraining the models requires the Teachable Machine web tool: create an *Image Project*, add the `plastic` and `metal` classes with
the training images listed in `training_data.md`, keep the default advanced settings (epochs 50, batch size 16, learning rate 0.001),
train, and export the model as *TensorFlow > Keras*. Training in Teachable Machine is not seeded, so retrained models can differ slightly.

## Expected results (100 fixed test images; one training run per model)

**RQ1: training data size**

| Model | Training images (plastic / metal) | Accuracy | Plastic recall | Metal recall | Metal precision | Metal F1 | TP/FP/FN/TN (metal = positive) |
|---|---|---|---|---|---|---|---|
| M1 | 10 / 10 | 0.87 | 0.94 | 0.80 | 0.93 | 0.86 | 40/3/10/47 |
| M2 | 30 / 30 | 0.91 | 0.88 | 0.94 | 0.89 | 0.91 | 47/6/3/44 |
| M4 | 100 / 100 | 0.91 | 0.86 | 0.96 | 0.87 | 0.91 | 48/7/2/43 |
| M5 | 300 / 300 | 0.95 | 0.92 | 0.98 | 0.92 | 0.95 | 49/4/1/46 |

**RQ2: class imbalance** (plastic fixed at 100 training images)

| Model | Training images (plastic / metal) | Accuracy | Plastic recall | Metal recall | Metal precision | Metal F1 | TP/FP/FN/TN (metal = positive) |
|---|---|---|---|---|---|---|---|
| B1 (= M4) | 100 / 100 | 0.91 | 0.86 | 0.96 | 0.87 | 0.91 | 48/7/2/43 |
| B2 | 100 / 20 | 0.84 | 1.00 | 0.68 | 1.00 | 0.81 | 34/0/16/50 |
| B3 | 100 / 5 | 0.58 | 1.00 | 0.16 | 1.00 | 0.28 | 8/0/42/50 |

M3 is excluded because its training images are the test images.

## Limitations

- The test set has only 100 images (one image = 1 percentage point), and each configuration was trained once; small differences are within noise.
- Teachable Machine training is not seeded or fully controllable, so exact numbers may vary when retraining.
- The training images were sampled at random and the exact file lists were not recorded (see `training_data.md`).
