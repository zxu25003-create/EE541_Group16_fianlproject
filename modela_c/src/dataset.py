import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import librosa

from augment import WaveformAugment     
from specAugment import SpecAugment      

class UrbanSoundMelDataset(Dataset):
    """
    UrbanSound8K -> log-mel spectrogram Dataset
    use_augment=True for Model A
    use_augment=False for Model C
    """
    def __init__(
        self,
        csv_path: str,
        audio_base_path: str,
        folds: list,
        sample_rate: int = 22050,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 512,
        use_augment: bool = False,
        waveform_augment: WaveformAugment | None = None,
        spec_augment: SpecAugment | None = None,
        clip_duration: float = 4.0,
    ):
        self.meta = pd.read_csv(csv_path)
        self.meta = self.meta[self.meta["fold"].isin(folds)].reset_index(drop=True)

        self.audio_base_path = audio_base_path
        self.sr = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.use_augment = use_augment
        self.waveform_augment = waveform_augment
        self.spec_augment = spec_augment
        self.clip_duration = clip_duration

    def __len__(self):
        return len(self.meta)

    def _load_waveform(self, idx):
        row = self.meta.iloc[idx]
        fold = row["fold"]
        file_name = row["slice_file_name"]
        file_path = os.path.join(self.audio_base_path, f"fold{fold}", file_name)

        y, sr = librosa.load(file_path, sr=self.sr, mono=True)

        
        target_len = int(self.clip_duration * self.sr)

        if len(y) < target_len:
            #not enough, pad zeros
            pad_width = target_len - len(y)
            y = np.pad(y, (0, pad_width), mode="constant")
        elif len(y) > target_len:
            # too long, cut
            if self.use_augment:
                # training: random crop
                max_start = len(y) - target_len
                start = np.random.randint(0, max_start + 1)
            else:
                # validation/test: center crop
                start = (len(y) - target_len) // 2
            y = y[start:start + target_len]

        # normalize waveform to zero mean and unit variance
        y = y - np.mean(y)
        std = np.std(y) + 1e-9
        y = y / std

        label = int(row["classID"])
        return y, label


    def _waveform_to_logmel(self, y_np: np.ndarray):
        mel = librosa.feature.melspectrogram(
            y=y_np,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        return log_mel  # [n_mels, T]

    def __getitem__(self, idx: int):
        # 1) read waveform + label
        y_np, label = self._load_waveform(idx)

        # 2) numpy -> tensor
        y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(0)  # (channels=1, samples=T)

        # 3) then waveform-level augmentation
        if self.use_augment and self.waveform_augment is not None:
            y = self.waveform_augment(y)

        # 4) waveform -> log-mel spectrogram
        y_np_aug = y.squeeze(0).cpu().numpy()
        log_mel = self._waveform_to_logmel(y_np_aug)
        log_mel = torch.tensor(log_mel, dtype=torch.float32)      # [n_mels, T]

        # 5) then spec-level augmentation
        if self.use_augment and self.spec_augment is not None:
            log_mel = self.spec_augment(log_mel)                 

        # 6) add channel dimension
        log_mel = log_mel.unsqueeze(0)

        label = torch.tensor(label, dtype=torch.long)
        return log_mel, label
