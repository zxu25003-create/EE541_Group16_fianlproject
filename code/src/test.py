

import torch
from augment import WaveformAugment

if __name__ == "__main__":
    sr = 22050
    t = torch.linspace(0, 1, steps=sr)
    x = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0)  # 440 Hz 正弦波

    aug = WaveformAugment(
        sample_rate=sr,
        max_shift_seconds=0.0, p_time_shift=0.0,
        gain_db_min=0.0, gain_db_max=0.0, p_gain=0.0,
        snr_db_min=30.0, snr_db_max=30.0, p_noise=0.0,
        pitch_semitones_min=0.0, pitch_semitones_max=0.0, p_pitch=0.0,
        stretch_min=0.9, stretch_max=1.1, p_stretch=1.0,
    )

    x_st = aug(x)
    print("Original shape:", x.shape)
    print("Stretched shape:", x_st.shape)
    print("First 10 samples (orig):", x[0, :10])
    print("First 10 samples (st):  ", x_st[0, :10])
