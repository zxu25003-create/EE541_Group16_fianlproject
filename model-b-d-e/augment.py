import random
import torch
import librosa  #for pitch shift
import numpy as np
class WaveformAugment:
    """Class for applying waveform augmentations."""
    def __init__(
            self,
            #--Time shift parameters--
            sample_rate:int,
            max_shift_seconds:float=0.5,
            p_time_shift:float=0.5,
            #--Random gain parameters--
            gain_db_min:float=-6.0,
            gain_db_max:float=6.0,
            p_gain:float=0.5,
            #--Additive noise parameters--
            snr_db_min:float=10.0,
            snr_db_max:float=30.0,
            p_noise:float=0.5,
            #--Pitch Shift--
            pitch_semitones_min:float=-2.0,
            pitch_semitones_max:float=2.0,
            p_pitch:float=0.3,
            #--Time Stretch--
            stretch_min:float=0.9,
            stretch_max:float=1.1,
            p_stretch:float=0.3,
    ):
        
        self.sample_rate = sample_rate
        #time shift params
        self.max_shift_seconds = max_shift_seconds
        self.p_time_shift = p_time_shift
        #random gain params
        self.gain_db_min = gain_db_min
        self.gain_db_max = gain_db_max
        self.p_gain = p_gain
        #additive noise params
        self.snr_db_min = snr_db_min
        self.snr_db_max = snr_db_max
        self.p_noise = p_noise
        #pitch shift params
        self.pitch_semitones_min = pitch_semitones_min
        self.pitch_semitones_max = pitch_semitones_max
        self.p_pitch = p_pitch
        #time stretch params
        self.stretch_min = stretch_min
        self.stretch_max = stretch_max
        self.p_stretch = p_stretch
    
    def __call__(self, waveform:torch.Tensor) -> torch.Tensor:
        """Apply augmentations to the waveform.

        Args:
            waveform (torch.Tensor): Input waveform tensor of shape (channels, samples).

        Returns:
            torch.Tensor: Augmented waveform tensor.
        """
        #for safety, clone the input waveform
        x = waveform.clone()
        # 1) Apply random time shift
        if random.random() < self.p_time_shift and self.p_time_shift > 0:
            x = self.random_time_shift(x)
        # 2) Apply random gain
        if self.p_gain>0 and random.random() < self.p_gain:
            x=self.random_gain(x)
        # 3)Additive noise(SNR)
        if self.p_noise>0 and random.random() < self.p_noise:
            x=self.add_noise(x)
        #4)Pitch shift
        if self.p_pitch>0 and random.random() < self.p_pitch:
            x = self.pitch_shift(x)
        #5)Time stretch
        if self.p_stretch>0 and random.random() < self.p_stretch:
            x = self.time_stretch(x)
        
        return x
    
    def random_time_shift(self, x:torch.Tensor) -> torch.Tensor:
            
        """Apply random time shift to the waveform.
        Args:
            x (torch.Tensor): Input waveform tensor of shape (channels, samples).
        Returns:
            torch.Tensor: Time-shifted waveform tensor.
        """
        #X` shape: (channels, samples)
        channels, num_samples = x.shape
        #MAX shift in samples
        max_shift_samples = int(self.max_shift_seconds * self.sample_rate)
        if max_shift_samples <= 0:
                return x
        shift=random.randint(-max_shift_samples, max_shift_samples)
        if shift == 0:
                return x
        x_shifted=torch.roll(x, shifts=shift, dims=-1)
        return x_shifted
    
    def random_gain(self, x:torch.Tensor) -> torch.Tensor:
        """Apply random gain to the waveform.
        Args:
            x (torch.Tensor): Input waveform tensor of shape (channels, samples).
        Returns:
            torch.Tensor: Gain-adjusted waveform tensor.
        """
        gain_db = random.uniform(self.gain_db_min, self.gain_db_max)
        gain = 10 ** (gain_db / 20)
        x = x * gain
        x= torch.clamp(x, -1.0, 1.0)  # Ensure waveform stays within valid range
        return x
    
    def add_noise(self, x:torch.Tensor) -> torch.Tensor:
        """Additive noise to the waveform based on SNR.
        Args:
            x (torch.Tensor): Input waveform tensor of shape (channels, samples).
        Returns:
            torch.Tensor: Noisy waveform tensor.
        """
        rms=x.pow(2).mean().sqrt()

        if rms<1e-6:
            return x
        
        snr_db=random.uniform(self.snr_db_min, self.snr_db_max)
        snr_linear=10**(snr_db/10)

        noise_rms=rms/ (snr_linear**0.5)
        noise=torch.randn_like(x) * noise_rms
        x_noisy=x + noise
        x_noisy=torch.clamp(x_noisy, -1.0, 1.0)  # Ensure waveform stays within valid range
        return x_noisy
    
    def pitch_shift(self, x:torch.Tensor) -> torch.Tensor:
        """Apply pitch shift to the waveform.
        Args:
            x (torch.Tensor): Input waveform tensor of shape (channels, samples).
        Returns:
            torch.Tensor: Pitch-shifted waveform tensor.
        """
        if x.dim() != 2 or x.shape[0] != 1:
            #raise ValueError("Input waveform must be mono with shape (1, samples).")
            return x
        
        n_steps = random.uniform(self.pitch_semitones_min, self.pitch_semitones_max)
        # If n_steps is very small, return the original waveform
        if abs(n_steps) < 1e-3:
            return x
        
        #tensor to numpy
        x_np = x.squeeze(0).cpu().numpy()#shape(samples,)

        #Apply pitch shift using librosa
        try:
            x_shifted_np = librosa.effects.pitch_shift(
                y=x_np,
                sr=self.sample_rate, 
                n_steps=n_steps)
        except Exception:
            #In case of any error during pitch shifting, return original waveform
            return x
        
        #make sure the output length matches input length
        if x_shifted_np.shape[0] != x_np.shape[0]:
            T=x_np.shape[0]
            if x_shifted_np.shape[0] > T:
                x_shifted_np = x_shifted_np[:T]
            else:
                pad_width = T - x_shifted_np.shape[0]
                x_shifted_np = np.pad(x_shifted_np, (0, pad_width), mode='edge')
        #numpy to tensor
        x_shifted = torch.from_numpy(x_shifted_np).unsqueeze(0).to(x.device, dtype=x.dtype)
        x_shifted = torch.clamp(x_shifted, -1.0, 1.0)  # Ensure waveform stays within valid range
        return x_shifted
    
    def time_stretch(self, x:torch.Tensor) -> torch.Tensor:
        """Apply time stretch to the waveform.
        Args:
            x (torch.Tensor): Input waveform tensor of shape (channels, samples).
        Returns:
            torch.Tensor: Time-stretched waveform tensor.
        """
        if x.dim() != 2 or x.shape[0] != 1:
            #raise ValueError("Input waveform must be mono with shape (1, samples).")
            return x
        
        rate = random.uniform(self.stretch_min, self.stretch_max)
        # If rate is very close to 1.0, return the original waveform
        if abs(rate - 1.0) < 1e-3:
            return x
        
        #tensor to numpy
        x_np = x.squeeze(0).cpu().numpy()#shape(samples,)

        #Apply time stretch using librosa
        try:
            x_stretched_np = librosa.effects.time_stretch(
                y=x_np,
                rate=rate)
        except Exception:
            #In case of any error during time stretching, return original waveform
            return x
        
        #make sure the output length matches input length
        if x_stretched_np.shape[0] != x_np.shape[0]:
            T=x_np.shape[0]
            if x_stretched_np.shape[0] > T:
                x_stretched_np = x_stretched_np[:T]
            else:
                pad_width = T - x_stretched_np.shape[0]
                x_stretched_np = np.pad(x_stretched_np, (0, pad_width), mode='edge')
        #numpy to tensor
        x_stretched = torch.from_numpy(x_stretched_np).unsqueeze(0).to(x.device, dtype=x.dtype)
        x_stretched = torch.clamp(x_stretched, -1.0, 1.0)  # Ensure waveform stays within valid range
        return x_stretched