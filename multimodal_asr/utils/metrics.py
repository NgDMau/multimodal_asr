"""
Evaluation metrics for ASR
"""

import torch
from typing import List, Dict, Tuple
import jiwer
import numpy as np
from collections import Counter
import re


def calculate_wer(predictions: List[str], references: List[str]) -> float:
    """
    Calculate Word Error Rate (WER)
    
    Args:
        predictions: List of predicted transcriptions
        references: List of reference transcriptions
        
    Returns:
        WER score (0.0 = perfect, higher = worse)
    """
    if len(predictions) != len(references):
        raise ValueError("Number of predictions and references must match")
    
    total_wer = 0.0
    for pred, ref in zip(predictions, references):
        # Normalize text
        pred_normalized = normalize_text_for_evaluation(pred)
        ref_normalized = normalize_text_for_evaluation(ref)
        
        # Calculate WER for this pair
        wer = jiwer.wer(ref_normalized, pred_normalized)
        total_wer += wer
    
    return total_wer / len(predictions)


def calculate_cer(predictions: List[str], references: List[str]) -> float:
    """
    Calculate Character Error Rate (CER)
    
    Args:
        predictions: List of predicted transcriptions
        references: List of reference transcriptions
        
    Returns:
        CER score (0.0 = perfect, higher = worse)
    """
    if len(predictions) != len(references):
        raise ValueError("Number of predictions and references must match")
    
    total_cer = 0.0
    for pred, ref in zip(predictions, references):
        # Normalize text
        pred_normalized = normalize_text_for_evaluation(pred)
        ref_normalized = normalize_text_for_evaluation(ref)
        
        # Calculate CER for this pair
        cer = jiwer.cer(ref_normalized, pred_normalized)
        total_cer += cer
    
    return total_cer / len(predictions)


def calculate_bleu(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Calculate BLEU scores (1-gram to 4-gram)
    
    Args:
        predictions: List of predicted transcriptions
        references: List of reference transcriptions
        
    Returns:
        Dictionary with BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
    """
    if len(predictions) != len(references):
        raise ValueError("Number of predictions and references must match")
    
    bleu_scores = {'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0}
    
    for pred, ref in zip(predictions, references):
        # Normalize text and tokenize
        pred_tokens = normalize_text_for_evaluation(pred).split()
        ref_tokens = normalize_text_for_evaluation(ref).split()
        
        # Calculate BLEU scores for different n-grams
        for n in range(1, 5):
            bleu_n = calculate_sentence_bleu(pred_tokens, [ref_tokens], n)
            bleu_scores[f'bleu_{n}'] += bleu_n
    
    # Average over all samples
    for key in bleu_scores:
        bleu_scores[key] /= len(predictions)
    
    return bleu_scores


def calculate_sentence_bleu(
    candidate: List[str], 
    references: List[List[str]], 
    n: int
) -> float:
    """
    Calculate sentence-level BLEU score for n-grams
    
    Args:
        candidate: Predicted sentence tokens
        references: List of reference sentence tokens
        n: N-gram order
        
    Returns:
        BLEU score for the sentence
    """
    if len(candidate) == 0:
        return 0.0
    
    # Get n-grams from candidate
    candidate_ngrams = get_ngrams(candidate, n)
    
    # Get maximum count for each n-gram across all references
    max_ref_counts = Counter()
    for ref in references:
        ref_ngrams = get_ngrams(ref, n)
        for ngram in ref_ngrams:
            max_ref_counts[ngram] = max(max_ref_counts[ngram], ref_ngrams[ngram])
    
    # Calculate precision
    clipped_count = 0
    total_count = len(candidate_ngrams)
    
    for ngram in candidate_ngrams:
        clipped_count += min(candidate_ngrams[ngram], max_ref_counts[ngram])
    
    if total_count == 0:
        return 0.0
    
    precision = clipped_count / total_count
    
    # Calculate brevity penalty
    candidate_length = len(candidate)
    closest_ref_length = min(len(ref) for ref in references)
    
    if candidate_length > closest_ref_length:
        brevity_penalty = 1.0
    else:
        brevity_penalty = np.exp(1 - closest_ref_length / candidate_length)
    
    # BLEU score
    bleu = brevity_penalty * precision
    return bleu


def get_ngrams(tokens: List[str], n: int) -> Counter:
    """
    Get n-grams from a list of tokens
    
    Args:
        tokens: List of tokens
        n: N-gram order
        
    Returns:
        Counter of n-grams
    """
    ngrams = Counter()
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i + n])
        ngrams[ngram] += 1
    return ngrams


def normalize_text_for_evaluation(text: str) -> str:
    """
    Normalize text for evaluation metrics
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove punctuation (optional, depending on evaluation protocol)
    text = re.sub(r'[^\w\s]', '', text)
    
    return text


def calculate_accuracy_metrics(
    predictions: List[str], 
    references: List[str]
) -> Dict[str, float]:
    """
    Calculate various accuracy metrics
    
    Args:
        predictions: List of predicted transcriptions
        references: List of reference transcriptions
        
    Returns:
        Dictionary with various metrics
    """
    metrics = {}
    
    # Word Error Rate
    metrics['wer'] = calculate_wer(predictions, references)
    
    # Character Error Rate
    metrics['cer'] = calculate_cer(predictions, references)
    
    # BLEU scores
    bleu_scores = calculate_bleu(predictions, references)
    metrics.update(bleu_scores)
    
    # Exact match accuracy
    exact_matches = sum(1 for pred, ref in zip(predictions, references) 
                       if normalize_text_for_evaluation(pred) == normalize_text_for_evaluation(ref))
    metrics['exact_match'] = exact_matches / len(predictions)
    
    return metrics


def calculate_perplexity(log_probs: torch.Tensor, lengths: torch.Tensor) -> float:
    """
    Calculate perplexity from log probabilities
    
    Args:
        log_probs: Log probabilities tensor (batch_size, seq_len, vocab_size)
        lengths: Actual sequence lengths (batch_size,)
        
    Returns:
        Perplexity score
    """
    # Sum log probabilities for each sequence
    total_log_prob = 0.0
    total_tokens = 0
    
    for i, length in enumerate(lengths):
        seq_log_probs = log_probs[i, :length]
        total_log_prob += seq_log_probs.sum().item()
        total_tokens += length.item()
    
    # Calculate average log probability
    avg_log_prob = total_log_prob / total_tokens
    
    # Perplexity is exp of negative average log probability
    perplexity = np.exp(-avg_log_prob)
    
    return perplexity


class MetricsTracker:
    """
    Class to track metrics during training and evaluation
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics"""
        self.predictions = []
        self.references = []
        self.losses = []
        self.total_samples = 0
    
    def update(
        self, 
        predictions: List[str], 
        references: List[str], 
        loss: float = None
    ):
        """
        Update metrics with new batch
        
        Args:
            predictions: Predicted transcriptions
            references: Reference transcriptions
            loss: Optional loss value
        """
        self.predictions.extend(predictions)
        self.references.extend(references)
        self.total_samples += len(predictions)
        
        if loss is not None:
            self.losses.append(loss)
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute all metrics from tracked data
        
        Returns:
            Dictionary with computed metrics
        """
        if len(self.predictions) == 0:
            return {}
        
        # Calculate ASR metrics
        metrics = calculate_accuracy_metrics(self.predictions, self.references)
        
        # Add loss if available
        if self.losses:
            metrics['avg_loss'] = np.mean(self.losses)
        
        # Add sample count
        metrics['num_samples'] = self.total_samples
        
        return metrics
    
    def get_sample_results(self, num_samples: int = 5) -> List[Dict[str, str]]:
        """
        Get sample predictions and references for inspection
        
        Args:
            num_samples: Number of samples to return
            
        Returns:
            List of dictionaries with prediction/reference pairs
        """
        samples = []
        for i in range(min(num_samples, len(self.predictions))):
            samples.append({
                'prediction': self.predictions[i],
                'reference': self.references[i]
            })
        return samples