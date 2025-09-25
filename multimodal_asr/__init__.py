"""
Multimodal ASR - A multimodal approach to Automatic Speech Recognition

This package provides tools and models for combining multiple modalities
(audio, text, and potentially visual information) for improved speech recognition.
"""

__version__ = "0.1.0"
__author__ = "NgDMau"

from .models import MultimodalASR
from .data import AudioProcessor, TextProcessor
from .utils import load_model, save_model

__all__ = [
    "MultimodalASR",
    "AudioProcessor", 
    "TextProcessor",
    "load_model",
    "save_model"
]