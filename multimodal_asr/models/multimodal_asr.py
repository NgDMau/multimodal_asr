"""
Main Multimodal ASR model combining all components
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple

from .audio_encoder import AudioEncoder
from .text_decoder import TextDecoder
from .fusion_module import FusionModule


class MultimodalASR(nn.Module):
    """
    Multimodal Automatic Speech Recognition model that combines:
    - Audio encoder for processing speech signals
    - Optional text context for better recognition
    - Optional visual features (lip reading, etc.)
    - Fusion module to combine modalities
    - Text decoder for generating transcriptions
    """
    
    def __init__(
        self,
        vocab_size: int,
        # Audio encoder parameters
        audio_input_dim: int = 80,
        audio_hidden_dim: int = 512,
        audio_num_layers: int = 6,
        audio_num_heads: int = 8,
        sample_rate: int = 16000,
        n_mels: int = 80,
        # Text decoder parameters
        text_hidden_dim: int = 512,
        text_num_layers: int = 6,
        text_num_heads: int = 8,
        max_text_length: int = 512,
        # Fusion parameters
        fusion_type: str = "attention",  # "concat", "attention", "cross_attention"
        use_text_context: bool = False,
        use_visual_features: bool = False,
        text_context_dim: int = 512,
        visual_dim: Optional[int] = None,
        # General parameters
        dropout: float = 0.1,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2
    ):
        super(MultimodalASR, self).__init__()
        
        self.vocab_size = vocab_size
        self.use_text_context = use_text_context
        self.use_visual_features = use_visual_features
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        
        # Audio encoder
        self.audio_encoder = AudioEncoder(
            input_dim=audio_input_dim,
            hidden_dim=audio_hidden_dim,
            num_layers=audio_num_layers,
            num_heads=audio_num_heads,
            dropout=dropout,
            sample_rate=sample_rate,
            n_mels=n_mels
        )
        
        # Text context encoder (if used)
        if use_text_context:
            self.text_context_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=text_context_dim,
                    nhead=text_num_heads,
                    dim_feedforward=text_context_dim * 4,
                    dropout=dropout,
                    activation='gelu',
                    batch_first=True
                ),
                num_layers=3
            )
            self.text_embedding = nn.Embedding(vocab_size, text_context_dim, padding_idx=pad_token_id)
        
        # Fusion module
        self.fusion_module = FusionModule(
            audio_dim=audio_hidden_dim,
            text_dim=text_context_dim if use_text_context else None,
            visual_dim=visual_dim if use_visual_features else None,
            hidden_dim=text_hidden_dim,  # Use text decoder's hidden dim as output
            fusion_type=fusion_type,
            num_heads=text_num_heads,
            dropout=dropout
        )
        
        # Text decoder
        self.text_decoder = TextDecoder(
            vocab_size=vocab_size,
            hidden_dim=text_hidden_dim,
            num_layers=text_num_layers,
            num_heads=text_num_heads,
            dropout=dropout,
            max_length=max_text_length,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id
        )
        
    def forward(
        self,
        audio: torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None,
        target_tokens: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
        text_context: Optional[torch.Tensor] = None,
        text_context_lengths: Optional[torch.Tensor] = None,
        visual_features: Optional[torch.Tensor] = None,
        visual_lengths: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the multimodal ASR model
        
        Args:
            audio: Raw audio tensor (batch_size, audio_length)
            audio_lengths: Lengths of audio sequences (batch_size,)
            target_tokens: Target transcription tokens for training (batch_size, text_length)
            target_lengths: Lengths of target sequences (batch_size,)
            text_context: Optional text context tokens (batch_size, context_length)
            text_context_lengths: Lengths of text context (batch_size,)
            visual_features: Optional visual features (batch_size, visual_length, visual_dim)
            visual_lengths: Lengths of visual sequences (batch_size,)
            
        Returns:
            Dictionary containing:
            - logits: Output logits (batch_size, text_length, vocab_size)
            - audio_features: Encoded audio features
            - attention_weights: Attention weights if available
        """
        batch_size = audio.size(0)
        
        # Encode audio
        audio_features, audio_mask = self.audio_encoder(audio, audio_lengths)
        
        # Process text context if provided
        text_features = None
        text_mask = None
        if self.use_text_context and text_context is not None:
            text_embeds = self.text_embedding(text_context)
            text_features = self.text_context_encoder(text_embeds)
            
            if text_context_lengths is not None:
                max_len = text_context.size(1)
                text_mask = torch.arange(max_len, device=text_context.device)[None, :] < text_context_lengths[:, None]
                text_mask = text_mask.float()
        
        # Process visual features if provided
        visual_mask = None
        if self.use_visual_features and visual_features is not None:
            if visual_lengths is not None:
                max_len = visual_features.size(1)
                visual_mask = torch.arange(max_len, device=visual_features.device)[None, :] < visual_lengths[:, None]
                visual_mask = visual_mask.float()
        
        # Fuse multimodal features
        fused_features = self.fusion_module(
            audio_features=audio_features,
            audio_mask=audio_mask,
            text_features=text_features,
            text_mask=text_mask,
            visual_features=visual_features,
            visual_mask=visual_mask
        )
        
        # Create target mask for decoder
        target_mask = None
        if target_lengths is not None and target_tokens is not None:
            max_len = target_tokens.size(1)
            target_mask = torch.arange(max_len, device=target_tokens.device)[None, :] < target_lengths[:, None]
            target_mask = target_mask.float()
        
        # Decode to text
        logits = self.text_decoder(
            encoder_outputs=fused_features,
            encoder_mask=audio_mask,  # Use audio mask as primary mask
            target_tokens=target_tokens,
            target_mask=target_mask
        )
        
        return {
            'logits': logits,
            'audio_features': audio_features,
            'fused_features': fused_features,
            'audio_mask': audio_mask
        }
    
    def generate(
        self,
        audio: torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None,
        text_context: Optional[torch.Tensor] = None,
        text_context_lengths: Optional[torch.Tensor] = None,
        visual_features: Optional[torch.Tensor] = None,
        visual_lengths: Optional[torch.Tensor] = None,
        max_length: int = 100,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: int = 50,
        top_p: float = 1.0,
        num_beams: int = 1
    ) -> Dict[str, torch.Tensor]:
        """
        Generate transcriptions for input audio
        
        Args:
            audio: Raw audio tensor (batch_size, audio_length)
            audio_lengths: Lengths of audio sequences (batch_size,)
            text_context: Optional text context tokens (batch_size, context_length)
            text_context_lengths: Lengths of text context (batch_size,)
            visual_features: Optional visual features (batch_size, visual_length, visual_dim)
            visual_lengths: Lengths of visual sequences (batch_size,)
            max_length: Maximum generation length
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            num_beams: Number of beams for beam search (not implemented yet)
            
        Returns:
            Dictionary containing generated tokens and scores
        """
        self.eval()
        
        with torch.no_grad():
            # Encode and fuse features
            outputs = self.forward(
                audio=audio,
                audio_lengths=audio_lengths,
                text_context=text_context,
                text_context_lengths=text_context_lengths,
                visual_features=visual_features,
                visual_lengths=visual_lengths
            )
            
            fused_features = outputs['fused_features']
            audio_mask = outputs['audio_mask']
            
            # Generate tokens
            generated_tokens = self.text_decoder.generate(
                encoder_outputs=fused_features,
                encoder_mask=audio_mask,
                max_length=max_length,
                temperature=temperature,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p
            )
            
        return {
            'generated_tokens': generated_tokens,
            'audio_features': outputs['audio_features'],
            'fused_features': fused_features
        }
    
    def get_model_size(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze_audio_encoder(self):
        """Freeze audio encoder parameters"""
        for param in self.audio_encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_audio_encoder(self):
        """Unfreeze audio encoder parameters"""
        for param in self.audio_encoder.parameters():
            param.requires_grad = True
    
    def freeze_text_decoder(self):
        """Freeze text decoder parameters"""
        for param in self.text_decoder.parameters():
            param.requires_grad = False
    
    def unfreeze_text_decoder(self):
        """Unfreeze text decoder parameters"""
        for param in self.text_decoder.parameters():
            param.requires_grad = True