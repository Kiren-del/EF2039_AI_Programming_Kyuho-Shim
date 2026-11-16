"""
Evaluate all Teachable Machine (Keras) models on one fixed test set.

Folder layout (relative to this script):
    dataset/plastic/*.jpg      fixed test images (never used for training)
    dataset/metal/*.jpg
    model/<NAME>/converted_keras/keras_model.h5
    model/<NAME>/converted_keras/labels.txt

Outputs (written to ./results):
    summary.csv                one row per model (accuracy, recall, precision, F1, TP/FP/FN/TN)
    predictions_<NAME>.csv     per-image predictions (for error analysis)
    confusion_matrices.png     confusion matrices of all models
    rq1_data_size.png          RQ1 plot (M1..M5)
    rq2_imbalance.png          RQ2 plot (B1(=M4), B2, B3)

Run:  python evaluate_models.py
Requires: tensorflow (or tf_keras), pillow, numpy, matplotlib
"""
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset"
MODEL_ROOT = ROOT / "model"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

# name -> description (training composition: plastic / metal)
MODELS = {
    "M1": "RQ1 10/10",
    "M2": "RQ1 30/30",
    "M3": "RQ1 50/50",
    "M4": "RQ1 100/100 (= B1)",
    "M5": "RQ1 300/300",
    "B2": "RQ2 100/20",
    "B3": "RQ2 100/5",
}
POSITIVE = "metal"  # positive class for precision / recall / F1

# Models whose training data overlaps the test set. Their scores are inflated, so they are
# flagged in summary.csv and excluded from the RQ1/RQ2 plots.
LEAKY = {"M3"}

# Training images per class, used as the x-axis of the RQ1 plot. EDIT if your counts differ.
SIZES = {"M1": 10, "M2": 30, "M3": 50, "M4": 100, "M5": 300}
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_keras_model(path):
    """Load a Teachable Machine .h5 model (handles the known DepthwiseConv2D 'groups' issue)."""
    try:
        import tensorflow as tf
        keras = tf.keras
    except ImportError:
        import tf_keras as keras  # fallback

    class DepthwiseConv2D(keras.layers.DepthwiseConv2D):
        def __init__(self, **kwargs):
            kwargs.pop("groups", None)
            super().__init__(**kwargs)

    try:
        return keras.models.load_model(path, compile=False)
    except Exception:
        return keras.models.load_model(
            path, compile=False, custom_objects={"DepthwiseConv2D": DepthwiseConv2D}
        )


def read_labels(path):
    labels = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            labels.append(line.split(" ", 1)[1] if " " in line else line)
    return labels


def preprocess(path):
    # same preprocessing as Teachable Machine's exported example code
    img = Image.open(path).convert("RGB")
    img = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32)
    return (arr / 127.5) - 1.0


def load_test_set(class_names):
    xs, ys, files = [], [], []
    for idx, name in enumerate(class_names):
        folder = DATASET / name
        paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTS)
        for p in paths:
            xs.append(preprocess(p))
            ys.append(idx)
            files.append(f"{name}/{p.name}")
    return np.stack(xs), np.array(ys), files


def confusion(y_true, y_pred, n):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm  # rows = true, cols = predicted


def main():
    first = MODEL_ROOT / "M1" / "converted_keras" / "labels.txt"
    class_names = read_labels(first)  # e.g. ['plastic', 'metal']
    x, y, files = load_test_set(class_names)
    counts = {n: int((y == i).sum()) for i, n in enumerate(class_names)}
    print(f"Test set: {len(y)} images {counts}")

    pos = class_names.index(POSITIVE)
    rows, all_cms = [], {}
    for name, desc in MODELS.items():
        mdir = MODEL_ROOT / name / "converted_keras"
        labels = read_labels(mdir / "labels.txt")
        if labels != class_names:
            sys.exit(f"[{name}] label order differs: {labels} vs {class_names}")
        model = load_keras_model(str(mdir / "keras_model.h5"))
        prob = model.predict(x, batch_size=32, verbose=0)
        pred = prob.argmax(axis=1)

        cm = confusion(y, pred, len(class_names))
        all_cms[name] = cm
        tp = cm[pos, pos]
        fn = cm[pos].sum() - tp
        fp = cm[:, pos].sum() - tp
        tn = cm.sum() - tp - fn - fp
        acc = np.trace(cm) / cm.sum()
        rec = tp / (tp + fn) if tp + fn else 0.0
        prec = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class_acc = {n: cm[i, i] / cm[i].sum() for i, n in enumerate(class_names)}

        rows.append({
            "model": name, "setting": desc,
            "test_set_note": "TRAIN-OVERLAP (exclude)" if name in LEAKY else "held-out",
            "accuracy": round(acc, 4),
            **{f"{n}_acc(recall)": round(per_class_acc[n], 4) for n in class_names},
            f"{POSITIVE}_precision": round(prec, 4),
            f"{POSITIVE}_f1": round(f1, 4),
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        })
        print(f"{name:3s} acc={acc:.3f} recall({POSITIVE})={rec:.3f} "
              f"prec({POSITIVE})={prec:.3f} f1={f1:.3f}  TP/FP/FN/TN={tp}/{fp}/{fn}/{tn}")

        with open(OUT / f"predictions_{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file", "true", "pred", *[f"prob_{n}" for n in class_names]])
            for fn_, t, p, pr in zip(files, y, pred, prob):
                w.writerow([fn_, class_names[t], class_names[p], *[f"{v:.4f}" for v in pr]])

    with open(OUT / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    make_plots(rows, all_cms, class_names)
    print(f"\nSaved results to: {OUT}")


def make_plots(rows, all_cms, class_names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # confusion matrices (models that overlap with the test set are left out), 2 x 3 grid
    shown = {k: v for k, v in all_cms.items() if k not in LEAKY}
    ncol = 3
    nrow = int(np.ceil(len(shown) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.3 * ncol, 2.3 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, cm) in zip(axes, shown.items()):
        ax.imshow(cm, cmap="Blues")
        ax.set_title(name, fontsize=10)
        ax.set_xticks(range(len(class_names)), class_names, fontsize=8)
        ax.set_yticks(range(len(class_names)), class_names, fontsize=8)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    for ax in axes[len(shown):]:
        ax.axis("off")
    fig.supxlabel("Predicted", fontsize=9)
    fig.supylabel("True", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrices.png", dpi=200)
    plt.close(fig)

    by = {r["model"]: r for r in rows}

    # RQ1: data size (images per class); models in LEAKY are skipped
    sizes = {m: s for m, s in SIZES.items() if m not in LEAKY and m in by}
    xs = [sizes[m] for m in sizes]
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(xs, [by[m]["accuracy"] for m in sizes], "o-", label="accuracy")
    for cname, style in zip(class_names, ["s--", "^:"]):
        ax.plot(xs, [by[m][f"{cname}_acc(recall)"] for m in sizes], style, label=f"{cname} recall")
    ax.set_xscale("log")
    ax.set_xticks(xs, [str(v) for v in xs])
    ax.minorticks_off()
    ax.set_xlabel("Training images per class")
    ax.set_ylabel("Score on fixed test set")
    ax.set_ylim(0.75, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "rq1_data_size.png", dpi=200)
    plt.close(fig)

    # RQ2: imbalance (metal training images: 100 / 20 / 5); legend placed below the axes
    order = [(m, lab) for m, lab in [("M4", "100:100"), ("B2", "100:20"), ("B3", "100:5")]
             if m not in LEAKY]
    fig, ax = plt.subplots(figsize=(4, 3.3))
    w = 0.25
    xx = np.arange(len(order))
    ax.bar(xx - w, [by[m]["accuracy"] for m, _ in order], w, label="accuracy")
    ax.bar(xx, [by[m][f"{POSITIVE}_acc(recall)"] for m, _ in order], w, label=f"{POSITIVE} recall")
    ax.bar(xx + w, [by[m][f"{POSITIVE}_precision"] for m, _ in order], w, label=f"{POSITIVE} precision")
    ax.set_xticks(xx, [lab for _, lab in order])
    ax.set_xlabel("Training ratio (plastic:metal)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "rq2_imbalance.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
