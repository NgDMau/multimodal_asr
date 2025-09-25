"""
Dataset classes for multimodal ASR
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple, Any
import os
import json
from .audio_processor import AudioProcessor
from .text_processor import TextProcessor


class MultimodalASRDataset(Dataset):
    """
    Dataset for multimodal ASR training and evaluation
    """
    
    def __init__(
        self,
        data_list: List[Dict[str, Any]],
        audio_processor: AudioProcessor,
        text_processor: TextProcessor,
        max_audio_length: Optional[int] = None,
        max_text_length: Optional[int] = None,
        use_text_context: bool = False,
        use_visual_features: bool = False
    ):
        """
        Initialize dataset
        
        Args:
            data_list: List of data samples, each containing:
                - 'audio_path': path to audio file
                - 'text': transcription text
                - 'text_context': optional text context
                - 'visual_features': optional visual features
            audio_processor: Audio preprocessing utility
            text_processor: Text preprocessing utility
            max_audio_length: Maximum audio length in samples
            max_text_length: Maximum text length in tokens
            use_text_context: Whether to use text context
            use_visual_features: Whether to use visual features
        """
        self.data_list = data_list
        self.audio_processor = audio_processor
        self.text_processor = text_processor
        self.max_audio_length = max_audio_length
        self.max_text_length = max_text_length
        self.use_text_context = use_text_context
        self.use_visual_features = use_visual_features
        
        # Filter valid samples
        self.valid_samples = self._filter_valid_samples()
        
    def _filter_valid_samples(self) -> List[Dict[str, Any]]:
        """Filter samples with valid audio and text"""
        valid_samples = []
        
        for sample in self.data_list:
            # Check if audio file exists
            if 'audio_path' in sample and os.path.exists(sample['audio_path']):
                # Check if text is available
                if 'text' in sample and sample['text'].strip():
                    valid_samples.append(sample)
        
        print(f"Found {len(valid_samples)} valid samples out of {len(self.data_list)}")
        return valid_samples
    
    def __len__(self) -> int:
        return len(self.valid_samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing processed sample data
        """
        sample = self.valid_samples[idx]
        
        # Load and process audio
        audio_path = sample['audio_path']
        waveform, _ = self.audio_processor.load_audio(audio_path)
        waveform = self.audio_processor.preprocess_audio(waveform)
        
        # Trim/pad audio if max length specified
        if self.max_audio_length:
            waveform = self.audio_processor.pad_or_trim(waveform, self.max_audio_length)
        
        audio_length = torch.tensor(waveform.shape[-1], dtype=torch.long)
        
        # Process text
        text = sample['text']
        token_ids = self.text_processor.encode(text, add_special_tokens=True)
        
        # Trim/pad text if max length specified
        if self.max_text_length and len(token_ids) > self.max_text_length:
            token_ids = token_ids[:self.max_text_length]
            # Ensure EOS token at end
            token_ids[-1] = self.text_processor.eos_token_id
        
        text_length = torch.tensor(len(token_ids), dtype=torch.long)
        token_ids = torch.tensor(token_ids, dtype=torch.long)
        
        result = {
            'audio': waveform,
            'audio_length': audio_length,
            'text_tokens': token_ids,
            'text_length': text_length,
            'text': text  # Keep original text for evaluation
        }
        
        # Add text context if available and requested
        if self.use_text_context and 'text_context' in sample:
            context_text = sample['text_context']
            context_token_ids = self.text_processor.encode(context_text, add_special_tokens=False)
            context_length = torch.tensor(len(context_token_ids), dtype=torch.long)
            context_token_ids = torch.tensor(context_token_ids, dtype=torch.long)
            
            result.update({
                'text_context': context_token_ids,
                'text_context_length': context_length
            })
        
        # Add visual features if available and requested
        if self.use_visual_features and 'visual_features' in sample:
            visual_features = sample['visual_features']
            if isinstance(visual_features, str):
                # Load visual features from file
                visual_features = torch.load(visual_features)
            else:
                visual_features = torch.tensor(visual_features, dtype=torch.float32)
            
            visual_length = torch.tensor(visual_features.shape[0], dtype=torch.long)
            
            result.update({
                'visual_features': visual_features,
                'visual_length': visual_length
            })
        
        return result


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Batched data dictionary
    """
    # Get batch size
    batch_size = len(batch)
    
    # Collect all audio waveforms and lengths
    audio_waveforms = [sample['audio'] for sample in batch]
    audio_lengths = torch.stack([sample['audio_length'] for sample in batch])
    
    # Pad audio to same length
    max_audio_length = max(waveform.shape[-1] for waveform in audio_waveforms)
    padded_audio = []
    for waveform in audio_waveforms:
        if waveform.shape[-1] < max_audio_length:
            pad_length = max_audio_length - waveform.shape[-1]
            padded_waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        else:
            padded_waveform = waveform
        padded_audio.append(padded_waveform)
    
    batched_audio = torch.stack(padded_audio)
    
    # Collect text tokens and lengths
    text_tokens_list = [sample['text_tokens'] for sample in batch]
    text_lengths = torch.stack([sample['text_length'] for sample in batch])
    
    # Pad text tokens to same length
    max_text_length = max(tokens.shape[-1] for tokens in text_tokens_list)
    padded_text_tokens = []
    for tokens in text_tokens_list:
        if tokens.shape[-1] < max_text_length:
            pad_length = max_text_length - tokens.shape[-1]
            padded_tokens = torch.nn.functional.pad(tokens, (0, pad_length), value=0)  # Pad with pad_token_id
        else:
            padded_tokens = tokens
        padded_text_tokens.append(padded_tokens)
    
    batched_text_tokens = torch.stack(padded_text_tokens)
    
    # Collect original texts for evaluation
    original_texts = [sample['text'] for sample in batch]
    
    result = {
        'audio': batched_audio,
        'audio_lengths': audio_lengths,
        'text_tokens': batched_text_tokens,
        'text_lengths': text_lengths,
        'original_texts': original_texts
    }
    
    # Handle text context if present
    if 'text_context' in batch[0]:
        text_context_list = [sample['text_context'] for sample in batch]
        text_context_lengths = torch.stack([sample['text_context_length'] for sample in batch])
        
        # Pad text context
        max_context_length = max(context.shape[-1] for context in text_context_list)
        padded_context = []
        for context in text_context_list:
            if context.shape[-1] < max_context_length:
                pad_length = max_context_length - context.shape[-1]
                padded_context_tokens = torch.nn.functional.pad(context, (0, pad_length), value=0)
            else:
                padded_context_tokens = context
            padded_context.append(padded_context_tokens)
        
        batched_context = torch.stack(padded_context)
        
        result.update({
            'text_context': batched_context,
            'text_context_lengths': text_context_lengths
        })
    
    # Handle visual features if present
    if 'visual_features' in batch[0]:
        visual_features_list = [sample['visual_features'] for sample in batch]
        visual_lengths = torch.stack([sample['visual_length'] for sample in batch])
        
        # Pad visual features
        max_visual_length = max(features.shape[0] for features in visual_features_list)
        visual_dim = visual_features_list[0].shape[-1]
        
        padded_visual = []
        for features in visual_features_list:
            if features.shape[0] < max_visual_length:
                pad_length = max_visual_length - features.shape[0]
                padded_features = torch.nn.functional.pad(
                    features, (0, 0, 0, pad_length), value=0
                )
            else:
                padded_features = features
            padded_visual.append(padded_features)
        
        batched_visual = torch.stack(padded_visual)
        
        result.update({
            'visual_features': batched_visual,
            'visual_lengths': visual_lengths
        })
    
    return result


def create_data_loader(
    dataset: MultimodalASRDataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create DataLoader for the dataset
    
    Args:
        dataset: MultimodalASR dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )


def load_data_from_json(json_file: str) -> List[Dict[str, Any]]:
    """
    Load dataset from JSON file
    
    Args:
        json_file: Path to JSON file containing dataset
        
    Returns:
        List of data samples
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def create_sample_dataset(
    audio_dir: str,
    transcripts_file: str,
    output_file: str
) -> None:
    """
    Create a sample dataset JSON file from audio directory and transcripts
    
    Args:
        audio_dir: Directory containing audio files
        transcripts_file: File containing transcriptions (format: filename\ttext)
        output_file: Output JSON file path
    """
    samples = []
    
    # Read transcripts
    transcripts = {}
    with open(transcripts_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                filename = parts[0]
                text = parts[1]
                transcripts[filename] = text
    
    # Create samples
    for filename, text in transcripts.items():
        audio_path = os.path.join(audio_dir, filename)
        if os.path.exists(audio_path):
            sample = {
                'audio_path': audio_path,
                'text': text
            }
            samples.append(sample)
    
    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    
    print(f"Created dataset with {len(samples)} samples and saved to {output_file}")