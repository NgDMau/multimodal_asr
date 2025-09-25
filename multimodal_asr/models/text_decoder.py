"""
Text decoder for generating text output from encoded features
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import math


class TextDecoder(nn.Module):
    """
    Transformer-based text decoder for generating transcriptions
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_length: int = 512,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2
    ):
        super(TextDecoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_token_id)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout, max_length)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output projection to vocabulary
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        encoder_outputs: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        target_tokens: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the text decoder
        
        Args:
            encoder_outputs: Encoded features from encoder (batch_size, enc_seq_len, hidden_dim)
            encoder_mask: Mask for encoder outputs (batch_size, enc_seq_len)
            target_tokens: Target token IDs for training (batch_size, dec_seq_len)
            target_mask: Mask for target tokens (batch_size, dec_seq_len)
            
        Returns:
            logits: Output logits (batch_size, dec_seq_len, vocab_size)
        """
        if target_tokens is None:
            # Inference mode - generate tokens autoregressively
            return self.generate(encoder_outputs, encoder_mask)
        
        # Training mode
        batch_size, dec_seq_len = target_tokens.shape
        
        # Create token embeddings
        token_embeds = self.token_embedding(target_tokens)
        token_embeds = token_embeds * math.sqrt(self.hidden_dim)
        
        # Add positional encoding
        token_embeds = self.pos_encoding(token_embeds)
        
        # Create causal mask for decoder
        causal_mask = self._generate_square_subsequent_mask(dec_seq_len)
        causal_mask = causal_mask.to(token_embeds.device)
        
        # Create key padding masks
        tgt_key_padding_mask = None
        if target_mask is not None:
            tgt_key_padding_mask = ~target_mask.bool()
            
        memory_key_padding_mask = None
        if encoder_mask is not None:
            memory_key_padding_mask = ~encoder_mask.bool()
        
        # Apply transformer decoder
        decoder_output = self.transformer_decoder(
            tgt=token_embeds,
            memory=encoder_outputs,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        # Project to vocabulary
        logits = self.output_projection(decoder_output)
        
        return logits
    
    def generate(
        self,
        encoder_outputs: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: int = 50,
        top_p: float = 1.0
    ) -> torch.Tensor:
        """
        Generate text autoregressively
        
        Args:
            encoder_outputs: Encoded features (batch_size, enc_seq_len, hidden_dim)
            encoder_mask: Mask for encoder outputs (batch_size, enc_seq_len)
            max_length: Maximum generation length
            temperature: Sampling temperature
            do_sample: Whether to use sampling or greedy decoding
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            
        Returns:
            generated_tokens: Generated token IDs (batch_size, generated_length)
        """
        if max_length is None:
            max_length = self.max_length
            
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        
        # Initialize with BOS token
        generated_tokens = torch.full(
            (batch_size, 1), 
            self.bos_token_id, 
            dtype=torch.long, 
            device=device
        )
        
        # Create memory key padding mask
        memory_key_padding_mask = None
        if encoder_mask is not None:
            memory_key_padding_mask = ~encoder_mask.bool()
        
        for step in range(max_length - 1):
            # Get current sequence length
            current_length = generated_tokens.size(1)
            
            # Create token embeddings for current sequence
            token_embeds = self.token_embedding(generated_tokens)
            token_embeds = token_embeds * math.sqrt(self.hidden_dim)
            
            # Add positional encoding
            token_embeds = self.pos_encoding(token_embeds)
            
            # Create causal mask
            causal_mask = self._generate_square_subsequent_mask(current_length)
            causal_mask = causal_mask.to(device)
            
            # Apply transformer decoder
            decoder_output = self.transformer_decoder(
                tgt=token_embeds,
                memory=encoder_outputs,
                tgt_mask=causal_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
            
            # Get logits for the last position
            last_logits = self.output_projection(decoder_output[:, -1, :])  # (batch_size, vocab_size)
            
            # Apply temperature
            if temperature != 1.0:
                last_logits = last_logits / temperature
            
            # Generate next token
            if do_sample:
                next_token = self._sample_next_token(last_logits, top_k, top_p)
            else:
                next_token = torch.argmax(last_logits, dim=-1, keepdim=True)
            
            # Append to generated sequence
            generated_tokens = torch.cat([generated_tokens, next_token], dim=1)
            
            # Check if all sequences have generated EOS token
            if torch.all(next_token.squeeze(-1) == self.eos_token_id):
                break
        
        return generated_tokens
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for decoder"""
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask
    
    def _sample_next_token(
        self, 
        logits: torch.Tensor, 
        top_k: int, 
        top_p: float
    ) -> torch.Tensor:
        """Sample next token using top-k and top-p sampling"""
        # Top-k sampling
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        # Top-p (nucleus) sampling
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Keep at least one token
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
        
        # Sample from the filtered distribution
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)