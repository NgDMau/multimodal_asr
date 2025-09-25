"""
Text preprocessing utilities for multimodal ASR
"""

import torch
import re
import string
from typing import List, Dict, Optional, Tuple, Union
from collections import Counter
import json


class TextProcessor:
    """
    Text preprocessing and tokenization utilities for ASR
    """
    
    def __init__(
        self,
        vocab_file: Optional[str] = None,
        vocab_size: int = 10000,
        min_freq: int = 2,
        special_tokens: Optional[Dict[str, int]] = None,
        lowercase: bool = True,
        remove_punctuation: bool = False,
        normalize_whitespace: bool = True
    ):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.normalize_whitespace = normalize_whitespace
        
        # Special tokens
        if special_tokens is None:
            self.special_tokens = {
                '<pad>': 0,
                '<bos>': 1,
                '<eos>': 2,
                '<unk>': 3
            }
        else:
            self.special_tokens = special_tokens
        
        self.pad_token_id = self.special_tokens['<pad>']
        self.bos_token_id = self.special_tokens['<bos>']
        self.eos_token_id = self.special_tokens['<eos>']
        self.unk_token_id = self.special_tokens['<unk>']
        
        # Vocabulary mappings
        self.token_to_id = {}
        self.id_to_token = {}
        
        # Initialize with special tokens
        for token, idx in self.special_tokens.items():
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
        
        # Load vocabulary if provided
        if vocab_file:
            self.load_vocabulary(vocab_file)
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text for ASR processing
        
        Args:
            text: Input text string
            
        Returns:
            Normalized text
        """
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove extra whitespace
        if self.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove punctuation if specified
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Additional normalization for ASR
        # Expand contractions
        text = self._expand_contractions(text)
        
        # Normalize numbers to words (basic)
        text = self._normalize_numbers(text)
        
        return text
    
    def _expand_contractions(self, text: str) -> str:
        """Expand common English contractions"""
        contractions = {
            "don't": "do not",
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am",
            "'s": " is"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        return text
    
    def _normalize_numbers(self, text: str) -> str:
        """Basic number normalization (can be extended)"""
        # Simple digit to word mapping for common cases
        digit_words = {
            '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
            '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
        }
        
        # Replace single digits with words
        for digit, word in digit_words.items():
            text = re.sub(r'\b' + digit + r'\b', word, text)
        
        return text
    
    def build_vocabulary(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts
        
        Args:
            texts: List of text strings to build vocabulary from
        """
        # Collect all tokens
        token_counts = Counter()
        
        for text in texts:
            normalized_text = self.normalize_text(text)
            tokens = self.tokenize(normalized_text, use_vocab=False)
            token_counts.update(tokens)
        
        # Select tokens based on frequency
        selected_tokens = []
        for token, count in token_counts.most_common():
            if count >= self.min_freq and len(selected_tokens) < (self.vocab_size - len(self.special_tokens)):
                selected_tokens.append(token)
        
        # Build vocabulary mappings
        current_id = len(self.special_tokens)
        for token in selected_tokens:
            if token not in self.token_to_id:
                self.token_to_id[token] = current_id
                self.id_to_token[current_id] = token
                current_id += 1
        
        print(f"Built vocabulary with {len(self.token_to_id)} tokens")
    
    def tokenize(self, text: str, use_vocab: bool = True) -> List[str]:
        """
        Tokenize text into tokens
        
        Args:
            text: Input text string
            use_vocab: Whether to use vocabulary mapping
            
        Returns:
            List of tokens
        """
        # Simple whitespace tokenization (can be extended with more sophisticated methods)
        normalized_text = self.normalize_text(text)
        tokens = normalized_text.split()
        
        # Filter tokens if using vocabulary
        if use_vocab and self.token_to_id:
            filtered_tokens = []
            for token in tokens:
                if token in self.token_to_id:
                    filtered_tokens.append(token)
                else:
                    filtered_tokens.append('<unk>')
            tokens = filtered_tokens
        
        return tokens
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs
        
        Args:
            text: Input text string
            add_special_tokens: Whether to add BOS/EOS tokens
            
        Returns:
            List of token IDs
        """
        tokens = self.tokenize(text)
        
        # Convert tokens to IDs
        token_ids = []
        if add_special_tokens:
            token_ids.append(self.bos_token_id)
        
        for token in tokens:
            token_id = self.token_to_id.get(token, self.unk_token_id)
            token_ids.append(token_id)
        
        if add_special_tokens:
            token_ids.append(self.eos_token_id)
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded text string
        """
        tokens = []
        
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                
                # Skip special tokens if requested
                if skip_special_tokens and token in self.special_tokens:
                    continue
                
                tokens.append(token)
        
        return ' '.join(tokens)
    
    def batch_encode(
        self, 
        texts: List[str],
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
        add_special_tokens: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Batch encode multiple texts
        
        Args:
            texts: List of text strings
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            truncation: Whether to truncate long sequences
            add_special_tokens: Whether to add special tokens
            
        Returns:
            Dictionary with input_ids and attention_mask tensors
        """
        # Encode all texts
        all_token_ids = []
        for text in texts:
            token_ids = self.encode(text, add_special_tokens=add_special_tokens)
            all_token_ids.append(token_ids)
        
        # Determine max length
        if max_length is None:
            max_length = max(len(token_ids) for token_ids in all_token_ids)
        
        # Pad/truncate sequences
        padded_token_ids = []
        attention_masks = []
        
        for token_ids in all_token_ids:
            # Truncate if necessary
            if truncation and len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
                # Ensure EOS token at end if truncated
                if add_special_tokens:
                    token_ids[-1] = self.eos_token_id
            
            # Create attention mask
            attention_mask = [1] * len(token_ids)
            
            # Pad if necessary
            if padding and len(token_ids) < max_length:
                pad_length = max_length - len(token_ids)
                token_ids.extend([self.pad_token_id] * pad_length)
                attention_mask.extend([0] * pad_length)
            
            padded_token_ids.append(token_ids)
            attention_masks.append(attention_mask)
        
        return {
            'input_ids': torch.tensor(padded_token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_masks, dtype=torch.long)
        }
    
    def batch_decode(
        self, 
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True
    ) -> List[str]:
        """
        Batch decode token ID tensors to texts
        
        Args:
            token_ids: Tensor of token IDs (batch_size, seq_length)
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            List of decoded text strings
        """
        batch_size = token_ids.size(0)
        decoded_texts = []
        
        for i in range(batch_size):
            token_id_list = token_ids[i].tolist()
            decoded_text = self.decode(token_id_list, skip_special_tokens=skip_special_tokens)
            decoded_texts.append(decoded_text)
        
        return decoded_texts
    
    def save_vocabulary(self, vocab_file: str) -> None:
        """Save vocabulary to file"""
        vocab_data = {
            'token_to_id': self.token_to_id,
            'special_tokens': self.special_tokens,
            'vocab_size': len(self.token_to_id)
        }
        
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        
        print(f"Vocabulary saved to {vocab_file}")
    
    def load_vocabulary(self, vocab_file: str) -> None:
        """Load vocabulary from file"""
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.token_to_id = vocab_data['token_to_id']
        self.special_tokens = vocab_data['special_tokens']
        
        # Rebuild id_to_token mapping
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        
        print(f"Vocabulary loaded from {vocab_file} with {len(self.token_to_id)} tokens")
    
    @property
    def vocabulary_size(self) -> int:
        """Get current vocabulary size"""
        return len(self.token_to_id)
    
    def get_token_frequency(self, texts: List[str]) -> Dict[str, int]:
        """Get token frequency statistics"""
        token_counts = Counter()
        
        for text in texts:
            tokens = self.tokenize(text, use_vocab=False)
            token_counts.update(tokens)
        
        return dict(token_counts)