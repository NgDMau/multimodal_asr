"""
Audio preprocessing utilities for multimodal ASR
"""

import torch
import torchaudio
import numpy as np
import librosa
from typing import Tuple, Optional, List, Union
import soundfile as sf


class AudioProcessor:
    """
    Audio preprocessing utilities for ASR
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        normalize: bool = True,
        augment: bool = False
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.normalize = normalize
        self.augment = augment
        
        # Audio transforms
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            f_min=0.0,
            f_max=sample_rate / 2,
            power=2.0,
            normalized=False
        )
        
        # Augmentation transforms (if enabled)
        if augment:
            self.time_stretch = torchaudio.transforms.TimeStretch()
            self.pitch_shift = torchaudio.transforms.PitchShift(sample_rate)
            self.add_noise = True
        
    def load_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """
        Load audio file and resample to target sample rate
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (waveform, sample_rate)
        """
        try:
            # Use torchaudio for loading
            waveform, orig_sr = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if necessary
            if orig_sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=orig_sr,
                    new_freq=self.sample_rate
                )
                waveform = resampler(waveform)
            
            return waveform.squeeze(0), self.sample_rate
            
        except Exception as e:
            # Fallback to librosa/soundfile
            try:
                waveform, sr = sf.read(audio_path)
                if len(waveform.shape) > 1:
                    waveform = np.mean(waveform, axis=1)
                
                if sr != self.sample_rate:
                    waveform = librosa.resample(
                        waveform, 
                        orig_sr=sr, 
                        target_sr=self.sample_rate
                    )
                
                return torch.FloatTensor(waveform), self.sample_rate
                
            except Exception as e2:
                raise RuntimeError(f"Failed to load audio file {audio_path}: {e2}")
    
    def preprocess_audio(
        self, 
        waveform: torch.Tensor,
        apply_augmentation: bool = None
    ) -> torch.Tensor:
        """
        Preprocess audio waveform
        
        Args:
            waveform: Audio waveform tensor (seq_length,)
            apply_augmentation: Whether to apply augmentation
            
        Returns:
            Preprocessed audio tensor
        """
        if apply_augmentation is None:
            apply_augmentation = self.augment
        
        # Ensure waveform is 1D
        if len(waveform.shape) > 1:
            waveform = waveform.squeeze()
        
        # Normalize amplitude
        if self.normalize:
            waveform = self._normalize_audio(waveform)
        
        # Apply augmentation if enabled
        if apply_augmentation and self.augment:
            waveform = self._apply_augmentation(waveform)
        
        return waveform
    
    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract mel-spectrogram features from audio
        
        Args:
            waveform: Audio waveform (seq_length,)
            
        Returns:
            Mel-spectrogram features (n_mels, time_frames)
        """
        # Ensure correct shape for mel transform
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
        
        # Extract mel-spectrogram
        mel_spec = self.mel_transform(waveform)
        
        # Convert to log scale
        log_mel = torch.log(mel_spec + 1e-6)
        
        return log_mel.squeeze(0)  # Remove batch dimension
    
    def _normalize_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """Normalize audio amplitude"""
        # Remove DC component
        waveform = waveform - torch.mean(waveform)
        
        # Normalize to [-1, 1] range
        max_val = torch.max(torch.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val
        
        return waveform
    
    def _apply_augmentation(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply audio augmentation"""
        if not self.augment:
            return waveform
        
        # Random selection of augmentations
        augmentations = []
        
        # Time stretching (speed perturbation)
        if np.random.random() < 0.3:
            rate = np.random.uniform(0.85, 1.15)
            try:
                waveform = librosa.effects.time_stretch(waveform.numpy(), rate=rate)
                waveform = torch.FloatTensor(waveform)
            except:
                pass  # Skip if fails
        
        # Add noise
        if np.random.random() < 0.3 and self.add_noise:
            noise_factor = np.random.uniform(0.001, 0.01)
            noise = torch.randn_like(waveform) * noise_factor
            waveform = waveform + noise
        
        # Volume perturbation
        if np.random.random() < 0.3:
            volume_factor = np.random.uniform(0.7, 1.3)
            waveform = waveform * volume_factor
            
        # Ensure normalized after augmentation
        waveform = self._normalize_audio(waveform)
        
        return waveform
    
    def pad_or_trim(
        self, 
        waveform: torch.Tensor, 
        target_length: int,
        pad_mode: str = 'constant'
    ) -> torch.Tensor:
        """
        Pad or trim waveform to target length
        
        Args:
            waveform: Input waveform
            target_length: Target length in samples
            pad_mode: Padding mode ('constant', 'reflect', 'replicate')
            
        Returns:
            Padded or trimmed waveform
        """
        current_length = waveform.shape[-1]
        
        if current_length > target_length:
            # Trim from center
            start = (current_length - target_length) // 2
            waveform = waveform[..., start:start + target_length]
        elif current_length < target_length:
            # Pad
            pad_length = target_length - current_length
            if pad_mode == 'constant':
                waveform = torch.nn.functional.pad(waveform, (0, pad_length), value=0)
            elif pad_mode == 'reflect':
                waveform = torch.nn.functional.pad(waveform, (0, pad_length), mode='reflect')
            elif pad_mode == 'replicate':
                waveform = torch.nn.functional.pad(waveform, (0, pad_length), mode='replicate')
        
        return waveform
    
    def compute_audio_lengths(self, waveforms: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute lengths of audio waveforms
        
        Args:
            waveforms: List of audio waveforms
            
        Returns:
            Tensor of lengths
        """
        lengths = [waveform.shape[-1] for waveform in waveforms]
        return torch.tensor(lengths, dtype=torch.long)
    
    def batch_process(
        self, 
        audio_paths: List[str],
        max_length: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a batch of audio files
        
        Args:
            audio_paths: List of paths to audio files
            max_length: Maximum length for padding/trimming
            
        Returns:
            Tuple of (batched_waveforms, lengths)
        """
        waveforms = []
        lengths = []
        
        for path in audio_paths:
            waveform, _ = self.load_audio(path)
            waveform = self.preprocess_audio(waveform)
            
            waveforms.append(waveform)
            lengths.append(waveform.shape[-1])
        
        # Determine max length for padding
        if max_length is None:
            max_length = max(lengths)
        
        # Pad all waveforms to same length
        padded_waveforms = []
        for waveform in waveforms:
            padded_waveform = self.pad_or_trim(waveform, max_length)
            padded_waveforms.append(padded_waveform)
        
        # Stack into batch
        batched_waveforms = torch.stack(padded_waveforms, dim=0)
        lengths_tensor = torch.tensor(lengths, dtype=torch.long)
        
        return batched_waveforms, lengths_tensor
    
    @staticmethod
    def get_audio_duration(waveform: torch.Tensor, sample_rate: int) -> float:
        """Get audio duration in seconds"""
        return waveform.shape[-1] / sample_rate