"""
Fusion module for combining multimodal inputs
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List


class FusionModule(nn.Module):
    """
    Fusion module for combining audio features with additional modalities
    like text context or visual features for improved ASR performance.
    """
    
    def __init__(
        self,
        audio_dim: int = 512,
        text_dim: int = 512,
        visual_dim: Optional[int] = None,
        hidden_dim: int = 512,
        fusion_type: str = "attention",  # "concat", "attention", "cross_attention"
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super(FusionModule, self).__init__()
        
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.visual_dim = visual_dim
        self.hidden_dim = hidden_dim
        self.fusion_type = fusion_type
        
        # Projection layers to common dimension
        self.audio_projection = nn.Linear(audio_dim, hidden_dim)
        
        if text_dim is not None:
            self.text_projection = nn.Linear(text_dim, hidden_dim)
        
        if visual_dim is not None:
            self.visual_projection = nn.Linear(visual_dim, hidden_dim)
        
        # Fusion-specific layers
        if fusion_type == "concat":
            # Simple concatenation followed by projection
            concat_dim = hidden_dim  # Start with audio
            if text_dim is not None:
                concat_dim += hidden_dim
            if visual_dim is not None:
                concat_dim += hidden_dim
            
            self.concat_projection = nn.Sequential(
                nn.Linear(concat_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            
        elif fusion_type == "attention":
            # Self-attention over concatenated modalities
            self.self_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            
        elif fusion_type == "cross_attention":
            # Cross-attention between modalities
            self.audio_text_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            
            if visual_dim is not None:
                self.audio_visual_attention = nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True
                )
                self.text_visual_attention = nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True
                )
        
        # Modality weights for adaptive fusion
        self.use_modality_weights = True
        if self.use_modality_weights:
            num_modalities = 1  # Audio is always present
            if text_dim is not None:
                num_modalities += 1
            if visual_dim is not None:
                num_modalities += 1
            
            self.modality_weights = nn.Parameter(torch.ones(num_modalities))
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        audio_features: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        visual_features: Optional[torch.Tensor] = None,
        visual_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of fusion module
        
        Args:
            audio_features: Audio features (batch_size, audio_seq_len, audio_dim)
            audio_mask: Audio attention mask (batch_size, audio_seq_len)
            text_features: Text features (batch_size, text_seq_len, text_dim)
            text_mask: Text attention mask (batch_size, text_seq_len)
            visual_features: Visual features (batch_size, visual_seq_len, visual_dim)
            visual_mask: Visual attention mask (batch_size, visual_seq_len)
            
        Returns:
            fused_features: Fused multimodal features (batch_size, audio_seq_len, hidden_dim)
        """
        # Project all modalities to common dimension
        audio_proj = self.audio_projection(audio_features)
        
        modality_features = [audio_proj]
        modality_masks = [audio_mask]
        
        if text_features is not None and hasattr(self, 'text_projection'):
            text_proj = self.text_projection(text_features)
            modality_features.append(text_proj)
            modality_masks.append(text_mask)
            
        if visual_features is not None and hasattr(self, 'visual_projection'):
            visual_proj = self.visual_projection(visual_features)
            modality_features.append(visual_proj)
            modality_masks.append(visual_mask)
        
        # Apply fusion strategy
        if self.fusion_type == "concat":
            fused_features = self._concat_fusion(modality_features, modality_masks)
            
        elif self.fusion_type == "attention":
            fused_features = self._attention_fusion(modality_features, modality_masks)
            
        elif self.fusion_type == "cross_attention":
            fused_features = self._cross_attention_fusion(
                audio_proj, text_proj if text_features is not None else None,
                visual_proj if visual_features is not None else None,
                audio_mask, text_mask, visual_mask
            )
        else:
            # Default to audio features only
            fused_features = audio_proj
        
        # Apply layer normalization and dropout
        fused_features = self.layer_norm(fused_features)
        fused_features = self.dropout(fused_features)
        
        return fused_features
    
    def _concat_fusion(
        self, 
        modality_features: List[torch.Tensor],
        modality_masks: List[Optional[torch.Tensor]]
    ) -> torch.Tensor:
        """Simple concatenation fusion"""
        # For different sequence lengths, we align to audio sequence length
        audio_features = modality_features[0]
        audio_seq_len = audio_features.size(1)
        
        # Interpolate other modalities to match audio sequence length
        aligned_features = [audio_features]
        
        for i, features in enumerate(modality_features[1:], 1):
            if features.size(1) != audio_seq_len:
                # Simple interpolation (could be improved with attention-based alignment)
                features = nn.functional.interpolate(
                    features.transpose(1, 2), 
                    size=audio_seq_len, 
                    mode='linear', 
                    align_corners=False
                ).transpose(1, 2)
            aligned_features.append(features)
        
        # Concatenate along feature dimension
        concatenated = torch.cat(aligned_features, dim=-1)
        
        # Project back to hidden dimension
        fused_features = self.concat_projection(concatenated)
        
        return fused_features
    
    def _attention_fusion(
        self,
        modality_features: List[torch.Tensor],
        modality_masks: List[Optional[torch.Tensor]]
    ) -> torch.Tensor:
        """Self-attention based fusion"""
        # Concatenate all modalities along sequence dimension
        all_features = []
        all_masks = []
        
        for i, (features, mask) in enumerate(zip(modality_features, modality_masks)):
            # Add modality type embedding (simple approach)
            modality_embed = torch.zeros_like(features[:, :1, :])
            modality_embed[:, :, i % features.size(-1)] = 1.0
            features_with_type = features + modality_embed
            
            all_features.append(features_with_type)
            if mask is not None:
                all_masks.append(mask)
        
        # Concatenate along sequence dimension
        concat_features = torch.cat(all_features, dim=1)
        
        if all_masks:
            concat_mask = torch.cat(all_masks, dim=1)
            key_padding_mask = ~concat_mask.bool()
        else:
            key_padding_mask = None
        
        # Apply self-attention
        attended_features, _ = self.self_attention(
            query=concat_features,
            key=concat_features,
            value=concat_features,
            key_padding_mask=key_padding_mask
        )
        
        # Return only the audio portion (first part of the sequence)
        audio_seq_len = modality_features[0].size(1)
        fused_features = attended_features[:, :audio_seq_len, :]
        
        return fused_features
    
    def _cross_attention_fusion(
        self,
        audio_features: torch.Tensor,
        text_features: Optional[torch.Tensor],
        visual_features: Optional[torch.Tensor],
        audio_mask: Optional[torch.Tensor],
        text_mask: Optional[torch.Tensor], 
        visual_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Cross-attention based fusion"""
        fused_features = audio_features
        
        # Audio-Text cross-attention
        if text_features is not None:
            text_key_padding_mask = ~text_mask.bool() if text_mask is not None else None
            
            attended_audio, _ = self.audio_text_attention(
                query=audio_features,
                key=text_features,
                value=text_features,
                key_padding_mask=text_key_padding_mask
            )
            fused_features = fused_features + attended_audio
        
        # Audio-Visual cross-attention
        if visual_features is not None and hasattr(self, 'audio_visual_attention'):
            visual_key_padding_mask = ~visual_mask.bool() if visual_mask is not None else None
            
            attended_audio_visual, _ = self.audio_visual_attention(
                query=fused_features,
                key=visual_features,
                value=visual_features,
                key_padding_mask=visual_key_padding_mask
            )
            fused_features = fused_features + attended_audio_visual
        
        return fused_features
    
    def get_modality_weights(self) -> torch.Tensor:
        """Get current modality weights"""
        if self.use_modality_weights:
            return torch.softmax(self.modality_weights, dim=0)
        else:
            return torch.ones(1)