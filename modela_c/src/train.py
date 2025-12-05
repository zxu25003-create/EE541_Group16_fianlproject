# train.py
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from augment import WaveformAugment
from specAugment import SpecAugment
from dataset import UrbanSoundMelDataset
from models import VGG13Mel

# --- for plotting ---
import matplotlib
matplotlib.use("Agg")  # 使用无界面后端，命令行/服务器环境也能画图
import matplotlib.pyplot as plt


# ------------------------- Fold split -------------------------
def build_folds(test_fold: int):
    """
    Given a test_fold (1~10), return train_folds, val_folds, test_folds
    val_fold is always the largest fold not used in test_fold
    """
    all_folds = list(range(1, 11))  # 1~10
    if test_fold not in all_folds:
        raise ValueError("test_fold must be in [1, 10]")

    remaining = [f for f in all_folds if f != test_fold]
    val_fold = max(remaining)
    train_folds = [f for f in remaining if f != val_fold]

    return train_folds, [val_fold], [test_fold]


# ------------------------- Dataloader -------------------------
def get_dataloaders(
    csv_path: str,
    audio_base_path: str,
    batch_size: int,
    model_type: str,   # "A" or "C"
    test_fold: int,
):
    train_folds, val_folds, test_folds = build_folds(test_fold)
    print(f"Using folds -> train: {train_folds}, val: {val_folds}, test: {test_folds}")

    if model_type == "A":
        use_augment = True
        waveform_augment = WaveformAugment(sample_rate=22050)
        spec_augment = SpecAugment()
    elif model_type == "C":
        # for control group, no augmentation
        use_augment = False
        waveform_augment = None
        spec_augment = None
    else:
        raise ValueError("model_type must be 'A' or 'C'")

    train_set = UrbanSoundMelDataset(
        csv_path=csv_path,
        audio_base_path=audio_base_path,
        folds=train_folds,
        use_augment=use_augment,
        waveform_augment=waveform_augment,
        spec_augment=spec_augment,
    )

    val_set = UrbanSoundMelDataset(
        csv_path=csv_path,
        audio_base_path=audio_base_path,
        folds=val_folds,
        use_augment=False,          # val/test without augment
        waveform_augment=None,
        spec_augment=None,
    )

    test_set = UrbanSoundMelDataset(
        csv_path=csv_path,
        audio_base_path=audio_base_path,
        folds=test_folds,
        use_augment=False,
        waveform_augment=None,
        spec_augment=None,
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return train_loader, val_loader, test_loader


# ------------------------- train / eval -------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_num = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_num += x.size(0)

    return total_loss / total_num, total_correct / total_num


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_num = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_num += x.size(0)

    return total_loss / total_num, total_correct / total_num


# ------------------------- plotting -------------------------
def plot_curves(epochs, train_losses, val_losses,
                train_accs, val_accs,
                save_dir, model_type, fold):
    """画 loss / acc 曲线，并保存到 save_dir 下。"""
    # Loss curve
    plt.figure()
    plt.plot(epochs, train_losses, label="train_loss")
    plt.plot(epochs, val_losses, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_type} Fold {fold} Loss")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(save_dir, f"{model_type}_fold{fold}_loss.png")
    plt.savefig(out_path)
    plt.close()

    # Accuracy curve
    plt.figure()
    plt.plot(epochs, train_accs, label="train_acc")
    plt.plot(epochs, val_accs, label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{model_type} Fold {fold} Accuracy")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(save_dir, f"{model_type}_fold{fold}_acc.png")
    plt.savefig(out_path)
    plt.close()


# ------------------------- main -------------------------
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
        "--model_type",
        type=str,
        choices=["A", "C"],
        required=True,
        help="A: with augmentation; C: control group without augmentation",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        help="which fold to use as test fold (1~10)",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="../checkpoints")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, test_loader = get_dataloaders(
        args.csv_path,
        args.audio_base_path,
        args.batch_size,
        args.model_type,
        args.fold,
    )

    model = VGG13Mel(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_val_acc = 0.0
    best_ckpt = os.path.join(
        args.save_dir, f"best_{args.model_type}_fold{args.fold}.pt"
    )

    # 用于画曲线的缓存
    epochs_list = []
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"[{args.model_type}] Fold {args.fold} Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.4f}"
        )

        # 记录当前 epoch 的指标
        epochs_list.append(epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # 更新 best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_ckpt)

    # test
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(
        f"[{args.model_type}] Fold {args.fold} TEST | "
        f"loss={test_loss:.4f} acc={test_acc:.4f}"
    )

    # 训练结束后自动画图
    plot_curves(
        epochs_list,
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        args.save_dir,
        args.model_type,
        args.fold,
    )


if __name__ == "__main__":
    main()
