import torch
import random

class SpecAugment:
    def __init__(
        self,
        time_mask_param: int = 20,   # every time mask can mask up to how many time steps
        freq_mask_param: int = 10,   # every freq mask can mask up to how many mel bins
        num_time_masks: int = 2,
        num_freq_masks: int = 2,
        p: float = 1.0,              # probability of applying SpecAugment
    ):
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks
        self.p = p

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        """
        spec: [n_mels, T] or [1, n_mels, T]
        return: same shape
        """
        if random.random() > self.p:
            return spec

        x = spec.clone()

        # Handle case where there is a channel dimension
        squeeze_channel = False
        if x.dim() == 3 and x.shape[0] == 1:
            x = x.squeeze(0)
            squeeze_channel = True

        n_mels, T = x.shape

        # freq masks
        for _ in range(self.num_freq_masks):
            f = random.randint(0, self.freq_mask_param)
            if f == 0:
                continue
            f0 = random.randint(0, max(0, n_mels - f))
            x[f0:f0+f, :] = 0.0

        # time masks
        for _ in range(self.num_time_masks):
            t = random.randint(0, self.time_mask_param)
            if t == 0:
                continue
            t0 = random.randint(0, max(0, T - t))
            x[:, t0:t0+t] = 0.0

        if squeeze_channel:
            x = x.unsqueeze(0)

        return x
