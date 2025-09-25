"""
Audio encoder for extracting features from audio input
"""

import torch
import torch.nn as nn
import torchaudio
from typing import Tuple, Optional


class AudioEncoder(nn.Module):
    """
    Audio encoder that processes raw audio and extracts meaningful features
    for speech recognition using convolutional layers and transformers.
    """
    
    def __init__(
        self,
        input_dim: int = 80,  # Number of mel-filterbank features
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        sample_rate: int = 16000,
        n_mels: int = 80
    ):
        super(AudioEncoder, self).__init__()
        
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        
        # Mel-spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=512,
            hop_length=160,
            win_length=400,
            f_min=0.0,
            f_max=sample_rate / 2,
            power=2.0,
            normalized=False
        )
        
        # Convolutional layers for local feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        # Calculate the output size after conv layers
        # After 2 stride-2 convolutions: n_mels // 4
        conv_output_dim = (n_mels // 4) * 128
        
        # Linear projection to hidden dimension
        self.feature_projection = nn.Linear(conv_output_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(
        self, 
        audio: torch.Tensor, 
        audio_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the audio encoder
        
        Args:
            audio: Raw audio tensor of shape (batch_size, seq_length)
            audio_lengths: Lengths of audio sequences (batch_size,)
            
        Returns:
            Tuple of (encoded_features, attention_mask)
            - encoded_features: (batch_size, time_steps, hidden_dim)
            - attention_mask: (batch_size, time_steps)
        """
        batch_size = audio.size(0)
        
        # Convert to mel-spectrogram
        # audio: (batch_size, seq_length) -> (batch_size, n_mels, time_frames)
        mel_spec = self.mel_transform(audio)
        
        # Add log and channel dimension for conv layers
        # (batch_size, n_mels, time_frames) -> (batch_size, 1, n_mels, time_frames)
        log_mel = torch.log(mel_spec + 1e-6).unsqueeze(1)
        
        # Apply convolutional layers
        # (batch_size, 1, n_mels, time_frames) -> (batch_size, 128, n_mels//4, time_frames//4)
        conv_features = self.conv_layers(log_mel)
        
        # Reshape for transformer: (batch_size, time_frames//4, n_mels//4 * 128)
        batch_size, channels, freq_bins, time_frames = conv_features.shape
        conv_features = conv_features.permute(0, 3, 2, 1)  # (batch, time, freq, channels)
        conv_features = conv_features.reshape(batch_size, time_frames, freq_bins * channels)
        
        # Project to hidden dimension
        features = self.feature_projection(conv_features)
        
        # Add positional encoding
        features = self.pos_encoding(features)
        
        # Create attention mask if lengths are provided
        attention_mask = None
        if audio_lengths is not None:
            # Calculate the corresponding time frames after downsampling
            frame_lengths = audio_lengths // (self.sample_rate // 100)  # Approximately 100 fps
            frame_lengths = frame_lengths // 4  # After 2 stride-2 convolutions
            max_frames = features.size(1)
            
            attention_mask = torch.arange(max_frames, device=features.device)[None, :] < frame_lengths[:, None]
            attention_mask = attention_mask.float()
        
        # Apply transformer encoder
        if attention_mask is not None:
            # Create key padding mask for transformer (True for padding positions)
            key_padding_mask = ~attention_mask.bool()
        else:
            key_padding_mask = None
            
        encoded_features = self.transformer(
            features, 
            src_key_padding_mask=key_padding_mask
        )
        
        # Apply layer normalization
        encoded_features = self.layer_norm(encoded_features)
        
        return encoded_features, attention_mask


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)