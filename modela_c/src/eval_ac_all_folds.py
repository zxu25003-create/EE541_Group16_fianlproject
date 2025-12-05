# eval_ac_all_folds.py
import os
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import UrbanSoundMelDataset
from models import VGG13Mel
from train import build_folds  # reuse the fold-split logic from train.py

# Visualization
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# For precision/recall/F1 and confusion matrix
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# UrbanSound8K class names in label-id order
CLASS_NAMES = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music",
]


@torch.no_grad()
def evaluate_with_preds(model, loader, criterion, device):
    """Evaluate model and also collect predictions/labels for metrics."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_num = 0

    all_labels = []
    all_preds = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_num += x.size(0)

        all_labels.append(y.cpu())
        all_preds.append(preds.cpu())

    labels = torch.cat(all_labels).numpy()
    preds = torch.cat(all_preds).numpy()

    return total_loss / total_num, total_correct / total_num, labels, preds


def get_test_loader(csv_path, audio_base_path, batch_size, test_fold):
    """Construct a test loader using only the specified test fold."""
    _, _, test_folds = build_folds(test_fold)
    test_set = UrbanSoundMelDataset(
        csv_path=csv_path,
        audio_base_path=audio_base_path,
        folds=test_folds,
        use_augment=False,      # absolutely no augmentation at test time
        waveform_augment=None,
        spec_augment=None,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=4
    )
    return test_loader


def plot_confusion_matrix(cm, class_names, title, save_path):
    """Plot a confusion matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_per_class_f1(f1_A, f1_C, class_names, save_path):
    """Plot per-class F1-score comparison between Model A and Model C."""
    x = np.arange(len(class_names))
    width = 0.35

    plt.figure(figsize=(8, 4))
    plt.bar(x - width / 2, f1_C, width, label="Model C")
    plt.bar(x + width / 2, f1_A, width, label="Model A")
    plt.xticks(x, class_names, rotation=45, ha="right")
    plt.ylabel("F1-score")
    plt.title("Per-class F1-score (Models A vs C)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path",
        type=str,
        default="../data/UrbanSound8K/metadata/UrbanSound8K.csv",
    )
    parser.add_argument(
        "--audio_base_path",
        type=str,
        default="../data/UrbanSound8K/audio",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="../checkpoints",
        help="Directory where A/C model checkpoints (.pt) are stored",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    criterion = nn.CrossEntropyLoss()

    # Store accuracy results, labels/preds, and per-class F1
    acc_results = {"A": None, "C": None}
    all_labels = {"A": [], "C": []}
    all_preds = {"A": [], "C": []}
    per_class_f1 = {}

    for model_type in ["C", "A"]:  # evaluate C first, then A
        print("=" * 60)
        print(f"Evaluating Model {model_type} on 10 folds")
        print("=" * 60)

        fold_accs = []

        for fold in range(1, 11):
            ckpt_path = os.path.join(
                args.save_dir, f"best_{model_type}_fold{fold}.pt"
            )
            if not os.path.exists(ckpt_path):
                print(f"[WARNING] checkpoint not found: {ckpt_path}")
                continue

            # Build test loader (current fold as test)
            test_loader = get_test_loader(
                args.csv_path,
                args.audio_base_path,
                args.batch_size,
                test_fold=fold,
            )

            # Build model and load weights for this fold
            model = VGG13Mel(num_classes=10).to(device)
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state)

            test_loss, test_acc, labels, preds = evaluate_with_preds(
                model, test_loader, criterion, device
            )
            fold_accs.append(test_acc)
            all_labels[model_type].append(labels)
            all_preds[model_type].append(preds)

            print(
                f"[{model_type}] Fold {fold} TEST | "
                f"loss={test_loss:.4f} acc={test_acc:.4f}"
            )

        fold_accs = np.array(fold_accs, dtype=float)
        acc_results[model_type] = fold_accs

        mean_acc = fold_accs.mean()
        std_acc = fold_accs.std()
        print(
            f"[{model_type}] 10-fold mean test acc = {mean_acc:.4f} "
            f"(std = {std_acc:.4f})"
        )

        # Concatenate predictions/labels over 10 folds for global metrics
        labels_all = np.concatenate(all_labels[model_type], axis=0)
        preds_all = np.concatenate(all_preds[model_type], axis=0)

        # Precision, recall, F1, and support (per class)
        precision, recall, f1, support = precision_recall_fscore_support(
            labels_all,
            preds_all,
            labels=np.arange(len(CLASS_NAMES)),
            zero_division=0,
        )
        per_class_f1[model_type] = f1

        # Save per-class metrics to CSV for reporting
        csv_path = os.path.join(args.save_dir, f"{model_type}_per_class_metrics.csv")
        with open(csv_path, "w") as f:
            f.write("class,precision,recall,f1,support\n")
            for c, name in enumerate(CLASS_NAMES):
                f.write(
                    f"{name},{precision[c]:.4f},{recall[c]:.4f},"
                    f"{f1[c]:.4f},{int(support[c])}\n"
                )
        print(f"Saved per-class metrics for Model {model_type} to {csv_path}")

        # Confusion matrix
        cm = confusion_matrix(
            labels_all,
            preds_all,
            labels=np.arange(len(CLASS_NAMES)),
        )
        cm_path = os.path.join(args.save_dir, f"{model_type}_confusion_matrix.png")
        plot_confusion_matrix(
            cm,
            CLASS_NAMES,
            title=f"Confusion Matrix - Model {model_type}",
            save_path=cm_path,
        )
        print(f"Saved confusion matrix figure for Model {model_type} to {cm_path}")

    # ---------- Figure 1: per-fold test accuracy (A vs C) ----------
    folds = np.arange(1, 11)

    plt.figure()
    width = 0.35
    plt.bar(folds - width / 2, acc_results["C"], width, label="Model C")
    plt.bar(folds + width / 2, acc_results["A"], width, label="Model A")
    plt.xlabel("Fold")
    plt.ylabel("Test Accuracy")
    plt.title("Per-fold Test Accuracy of Models A and C")
    plt.xticks(folds)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(args.save_dir, "AC_test_acc_per_fold.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved per-fold accuracy figure to {out_path}")

    # ---------- Figure 2: mean 10-fold accuracy (A vs C) ----------
    mean_C = acc_results["C"].mean()
    mean_A = acc_results["A"].mean()

    plt.figure()
    labels = ["Model C", "Model A"]
    means = [mean_C, mean_A]
    plt.bar(labels, means)
    plt.ylabel("Mean Test Accuracy")
    plt.title("10-fold Mean Test Accuracy (UrbanSound8K)")
    plt.tight_layout()
    out_path = os.path.join(args.save_dir, "AC_mean_test_acc.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved mean accuracy figure to {out_path}")

    # ---------- Figure 3: per-class F1-score (A vs C) ----------
    f1_C = per_class_f1["C"]
    f1_A = per_class_f1["A"]
    f1_path = os.path.join(args.save_dir, "AC_per_class_f1.png")
    plot_per_class_f1(f1_A, f1_C, CLASS_NAMES, f1_path)
    print(f"Saved per-class F1 figure to {f1_path}")


if __name__ == "__main__":
    main()
