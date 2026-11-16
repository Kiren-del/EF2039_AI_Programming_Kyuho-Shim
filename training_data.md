# Training data used for each model

All training images come from the Kaggle "Garbage classification" dataset (classes `plastic` and `metal`).

| Model | Plastic | Metal |
|---|---|---|
| M1 | 10 | 10 |
| M2 | 30 | 30 |
| M3 | 50 | 50 (same images as the test set; excluded from analysis) |
| M4 (= B1) | 100 | 100 |
| M5 | 300 | 300 |
| B2 | 100 | 20 |
| B3 | 100 | 5 |

## Test set

`plastic1-50` and `metal1-50` (see `dataset/test_split.csv`). The 50 metal test images were removed from the metal training pool,
which therefore consists of `metal51-410` (360 images).

## How the training images were chosen

- Training images were selected **at random** from the remaining pool (plastic: `plastic51-482`, metal: `metal51-410`);
  none of the plastic training images are among `plastic1-50`.
- Each model's training set was sampled **independently**; the smaller training sets are not subsets of the larger ones.
- The exact file lists were not recorded, so the training sets cannot be reconstructed exactly. M3 was trained on the 100 test images
  and is therefore not evaluated on held-out data.
