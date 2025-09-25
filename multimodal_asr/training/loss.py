"""
Loss functions for multimodal ASR training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class MultimodalLoss(nn.Module):
    """
    Combined loss function for multimodal ASR training
    """
    
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int = 0,
        label_smoothing: float = 0.0,
        ignore_index: int = -100,
        reduction: str = 'mean'
    ):
        super(MultimodalLoss, self).__init__()
        
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        self.reduction = reduction
        
        # Main cross-entropy loss with label smoothing
        self.cross_entropy = nn.CrossEntropyLoss(
            ignore_index=pad_token_id,
            label_smoothing=label_smoothing,
            reduction=reduction
        )
        
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss
        
        Args:
            logits: Model output logits (batch_size, seq_len, vocab_size)
            targets: Target token IDs (batch_size, seq_len)
            target_lengths: Actual target sequence lengths (batch_size,)
            
        Returns:
            Dictionary containing loss components
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # Flatten for cross-entropy computation
        logits_flat = logits.view(-1, vocab_size)  # (B*T, V)
        targets_flat = targets.view(-1)  # (B*T)
        
        # Compute main cross-entropy loss
        ce_loss = self.cross_entropy(logits_flat, targets_flat)
        
        # Create mask for valid positions if lengths provided
        loss_mask = None
        if target_lengths is not None:
            # Create mask: 1 for valid positions, 0 for padding
            max_len = targets.size(1)
            loss_mask = torch.arange(max_len, device=targets.device)[None, :] < target_lengths[:, None]
            loss_mask = loss_mask.float()
        
        loss_dict = {
            'total_loss': ce_loss,
            'ce_loss': ce_loss
        }
        
        # Add perplexity for monitoring
        with torch.no_grad():
            perplexity = torch.exp(ce_loss)
            loss_dict['perplexity'] = perplexity
        
        return loss_dict


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label smoothing cross-entropy loss
    """
    
    def __init__(
        self,
        smoothing: float = 0.1,
        ignore_index: int = -100,
        reduction: str = 'mean'
    ):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.reduction = reduction
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Apply label smoothing cross-entropy
        
        Args:
            logits: Model predictions (batch_size, num_classes)
            targets: Target labels (batch_size,)
            
        Returns:
            Loss tensor
        """
        vocab_size = logits.size(-1)
        
        # Create one-hot encoding
        true_dist = torch.zeros_like(logits)
        true_dist.fill_(self.smoothing / (vocab_size - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        # Mask ignore_index positions
        if self.ignore_index >= 0:
            mask = targets != self.ignore_index
            true_dist = true_dist * mask.unsqueeze(1).float()
        
        # Compute KL divergence
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -torch.sum(true_dist * log_probs, dim=-1)
        
        # Apply ignore mask
        if self.ignore_index >= 0:
            loss = loss * mask.float()
        
        # Reduction
        if self.reduction == 'mean':
            if self.ignore_index >= 0:
                return loss.sum() / mask.float().sum()
            else:
                return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        ignore_index: int = -100,
        reduction: str = 'mean'
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss
        
        Args:
            logits: Model predictions (batch_size, num_classes)
            targets: Target labels (batch_size,)
            
        Returns:
            Loss tensor
        """
        ce_loss = F.cross_entropy(
            logits, targets, 
            ignore_index=self.ignore_index, 
            reduction='none'
        )
        
        # Compute probabilities
        pt = torch.exp(-ce_loss)
        
        # Apply focal term
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # Reduction
        if self.reduction == 'mean':
            mask = targets != self.ignore_index
            return focal_loss.sum() / mask.float().sum()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CTCLoss(nn.Module):
    """
    CTC Loss for sequence-to-sequence training without alignment
    """
    
    def __init__(
        self,
        blank_token_id: int = 0,
        reduction: str = 'mean',
        zero_infinity: bool = False
    ):
        super(CTCLoss, self).__init__()
        self.blank = blank_token_id
        self.reduction = reduction
        self.zero_infinity = zero_infinity
        
        self.ctc_loss = nn.CTCLoss(
            blank=blank_token_id,
            reduction=reduction,
            zero_infinity=zero_infinity
        )
        
    def forward(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute CTC loss
        
        Args:
            log_probs: Log probabilities (seq_len, batch_size, num_classes)
            targets: Target sequences (batch_size, target_seq_len)
            input_lengths: Input sequence lengths (batch_size,)
            target_lengths: Target sequence lengths (batch_size,)
            
        Returns:
            CTC loss
        """
        # Flatten targets for CTC
        targets_concat = []
        for i, length in enumerate(target_lengths):
            targets_concat.extend(targets[i, :length].tolist())
        targets_concat = torch.tensor(targets_concat, device=targets.device)
        
        return self.ctc_loss(log_probs, targets_concat, input_lengths, target_lengths)


class CombinedLoss(nn.Module):
    """
    Combined loss function supporting multiple loss types
    """
    
    def __init__(
        self,
        loss_config: Dict[str, Any],
        vocab_size: int,
        pad_token_id: int = 0
    ):
        super(CombinedLoss, self).__init__()
        
        self.loss_weights = loss_config.get('weights', {'ce': 1.0})
        self.losses = nn.ModuleDict()
        
        # Initialize loss functions based on config
        if 'ce' in self.loss_weights:
            self.losses['ce'] = nn.CrossEntropyLoss(
                ignore_index=pad_token_id,
                label_smoothing=loss_config.get('label_smoothing', 0.0)
            )
        
        if 'focal' in self.loss_weights:
            self.losses['focal'] = FocalLoss(
                alpha=loss_config.get('focal_alpha', 1.0),
                gamma=loss_config.get('focal_gamma', 2.0),
                ignore_index=pad_token_id
            )
        
        if 'ctc' in self.loss_weights:
            self.losses['ctc'] = CTCLoss(
                blank_token_id=loss_config.get('ctc_blank', 0),
                zero_infinity=loss_config.get('ctc_zero_infinity', False)
            )
        
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss
        
        Args:
            logits: Model output logits
            targets: Target sequences
            input_lengths: Input sequence lengths (for CTC)
            target_lengths: Target sequence lengths
            
        Returns:
            Dictionary of loss components
        """
        loss_dict = {}
        total_loss = 0.0
        
        # Compute each loss component
        for loss_name, loss_fn in self.losses.items():
            if loss_name == 'ctc':
                # CTC requires special handling
                if input_lengths is not None and target_lengths is not None:
                    log_probs = F.log_softmax(logits, dim=-1)
                    log_probs = log_probs.transpose(0, 1)  # (T, B, V)
                    loss_value = loss_fn(log_probs, targets, input_lengths, target_lengths)
                else:
                    continue  # Skip CTC if lengths not provided
            else:
                # Standard losses
                batch_size, seq_len, vocab_size = logits.shape
                logits_flat = logits.view(-1, vocab_size)
                targets_flat = targets.view(-1)
                loss_value = loss_fn(logits_flat, targets_flat)
            
            # Weight and accumulate
            weighted_loss = self.loss_weights[loss_name] * loss_value
            loss_dict[f'{loss_name}_loss'] = loss_value
            total_loss += weighted_loss
        
        loss_dict['total_loss'] = total_loss
        
        return loss_dict