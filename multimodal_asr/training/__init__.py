"""
Training module for multimodal ASR
"""

from .trainer import Trainer
from .loss import MultimodalLoss

__all__ = [
    "Trainer",
    "MultimodalLoss"
]