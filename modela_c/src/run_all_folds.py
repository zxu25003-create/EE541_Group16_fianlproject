import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        choices=["A", "C"],
        required=True,
        help="A: with augmentation; C: control group without augmentation",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
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
    )
    args = parser.parse_args()

    python_exe = sys.executable
    src_dir = Path(__file__).parent

    for fold in range(1, 11):
        print("=" * 60)
        print(f"Running fold {fold} for model_type={args.model_type}")
        print("=" * 60)

        cmd = [
            python_exe,
            str(src_dir / "train.py"),
            "--model_type",
            args.model_type,
            "--fold",
            str(fold),
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--weight_decay",
            str(args.weight_decay),
            "--csv_path",
            args.csv_path,
            "--audio_base_path",
            args.audio_base_path,
            "--save_dir",
            args.save_dir,
        ]

        # run the command
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
