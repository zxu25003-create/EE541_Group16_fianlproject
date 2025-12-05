# test.py
import torch
from augment import WaveformAugment
from specAugment import SpecAugment
from dataset import UrbanSoundMelDataset
from models import VGG13Mel  

def main():
    csv_path = "../data/UrbanSound8K/metadata/UrbanSound8K.csv"
    audio_base = "../data/UrbanSound8K/audio"

    wave_aug = WaveformAugment(sample_rate=22050)
    spec_aug = SpecAugment()

    ds = UrbanSoundMelDataset(
        csv_path=csv_path,
        audio_base_path=audio_base,
        folds=[1],
        use_augment=True,
        waveform_augment=wave_aug,
        spec_augment=spec_aug,
    )

    x, y = ds[0]
    print("x shape:", x.shape)
    print("label:", y)

    x = x.unsqueeze(0)  # [1, 1, 128, 173]

    model = VGG13Mel(num_classes=10)
    logits = model(x)
    print("logits shape:", logits.shape)  # 理想是: torch.Size([1, 10])

if __name__ == "__main__":
    main()
