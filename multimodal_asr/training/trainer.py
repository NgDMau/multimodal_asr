"""
Training utilities for multimodal ASR
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import os
import time
from typing import Dict, Any, Optional, List, Tuple
import logging
from tqdm import tqdm

from ..models.multimodal_asr import MultimodalASR
from ..data.text_processor import TextProcessor
from ..utils.metrics import MetricsTracker, calculate_accuracy_metrics
from ..utils.model_utils import save_model
from .loss import MultimodalLoss


class Trainer:
    """
    Trainer class for multimodal ASR model
    """
    
    def __init__(
        self,
        model: MultimodalASR,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader],
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        loss_fn: MultimodalLoss,
        text_processor: TextProcessor,
        config: Any,
        device: torch.device,
        logger: Optional[logging.Logger] = None
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.text_processor = text_processor
        self.config = config
        self.device = device
        self.logger = logger or self._create_logger()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')  # For WER (lower is better)
        self.patience_counter = 0
        
        # Move model to device
        self.model.to(device)
        
        # Metrics tracking
        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()
        
        # Create output directories
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        
    def _create_logger(self) -> logging.Logger:
        """Create default logger"""
        logger = logging.getLogger('multimodal_asr_trainer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop
        
        Returns:
            Training history and metrics
        """
        self.logger.info(f"Starting training for {self.config.training.num_epochs} epochs")
        self.logger.info(f"Training on device: {self.device}")
        
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_wer': [],
            'learning_rates': []
        }
        
        for epoch in range(self.config.training.num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self._train_epoch()
            training_history['train_loss'].append(train_metrics['avg_loss'])
            
            # Validation phase
            if self.val_dataloader is not None:
                val_metrics = self._validate_epoch()
                training_history['val_loss'].append(val_metrics.get('avg_loss', 0))
                training_history['val_wer'].append(val_metrics.get('wer', 0))
                
                # Check for early stopping
                if self._should_early_stop(val_metrics):
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    metric = val_metrics.get('wer', train_metrics['avg_loss'])
                    self.scheduler.step(metric)
                else:
                    self.scheduler.step()
            
            # Record learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            training_history['learning_rates'].append(current_lr)
            
            # Save checkpoint
            if (epoch + 1) % self.config.training.save_steps == 0:
                self._save_checkpoint(epoch, val_metrics if self.val_dataloader else train_metrics)
        
        self.logger.info("Training completed")
        return training_history
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        self.train_metrics.reset()
        
        epoch_loss = 0.0
        num_batches = len(self.train_dataloader)
        
        progress_bar = tqdm(
            self.train_dataloader, 
            desc=f"Epoch {self.current_epoch + 1}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            loss = self._train_step(batch)
            epoch_loss += loss
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss:.4f}',
                'avg_loss': f'{epoch_loss / (batch_idx + 1):.4f}'
            })
            
            # Logging
            if (self.global_step + 1) % self.config.training.logging_steps == 0:
                self.logger.info(
                    f"Step {self.global_step + 1}, Loss: {loss:.4f}, "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                )
        
        avg_loss = epoch_loss / num_batches
        metrics = {'avg_loss': avg_loss}
        
        self.logger.info(
            f"Epoch {self.current_epoch + 1} completed. "
            f"Average loss: {avg_loss:.4f}"
        )
        
        return metrics
    
    def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step"""
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(
            audio=batch['audio'],
            audio_lengths=batch['audio_lengths'],
            target_tokens=batch['text_tokens'],
            target_lengths=batch['text_lengths']
        )
        
        # Compute loss
        loss_dict = self.loss_fn(
            logits=outputs['logits'],
            targets=batch['text_tokens'],
            target_lengths=batch['text_lengths']
        )
        
        loss = loss_dict['total_loss']
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if self.config.training.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.gradient_clip_norm
            )
        
        # Update parameters
        self.optimizer.step()
        self.global_step += 1
        
        return loss.item()
    
    def _validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        if self.val_dataloader is None:
            return {}
        
        self.model.eval()
        self.val_metrics.reset()
        
        epoch_loss = 0.0
        predictions = []
        references = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation", leave=False):
                batch = self._move_batch_to_device(batch)
                
                # Forward pass for loss
                outputs = self.model(
                    audio=batch['audio'],
                    audio_lengths=batch['audio_lengths'],
                    target_tokens=batch['text_tokens'],
                    target_lengths=batch['text_lengths']
                )
                
                # Compute loss
                loss_dict = self.loss_fn(
                    logits=outputs['logits'],
                    targets=batch['text_tokens'],
                    target_lengths=batch['text_lengths']
                )
                
                epoch_loss += loss_dict['total_loss'].item()
                
                # Generate predictions for metrics
                generated_outputs = self.model.generate(
                    audio=batch['audio'],
                    audio_lengths=batch['audio_lengths'],
                    max_length=100,
                    do_sample=False
                )
                
                # Decode predictions
                batch_predictions = self.text_processor.batch_decode(
                    generated_outputs['generated_tokens'],
                    skip_special_tokens=True
                )
                
                predictions.extend(batch_predictions)
                references.extend(batch['original_texts'])
        
        # Compute metrics
        avg_loss = epoch_loss / len(self.val_dataloader)
        accuracy_metrics = calculate_accuracy_metrics(predictions, references)
        
        metrics = {
            'avg_loss': avg_loss,
            **accuracy_metrics
        }
        
        self.logger.info(
            f"Validation - Loss: {avg_loss:.4f}, "
            f"WER: {accuracy_metrics['wer']:.4f}, "
            f"CER: {accuracy_metrics['cer']:.4f}"
        )
        
        return metrics
    
    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch tensors to device"""
        moved_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved_batch[key] = value.to(self.device)
            else:
                moved_batch[key] = value
        return moved_batch
    
    def _should_early_stop(self, val_metrics: Dict[str, float]) -> bool:
        """Check if training should stop early"""
        if not hasattr(self.config.training, 'early_stopping_patience'):
            return False
        
        metric_name = getattr(self.config.training, 'early_stopping_metric', 'wer')
        current_metric = val_metrics.get(metric_name, float('inf'))
        
        # For WER, lower is better
        if metric_name == 'wer' and current_metric < self.best_metric:
            self.best_metric = current_metric
            self.patience_counter = 0
            return False
        # For loss, lower is better
        elif metric_name == 'loss' and current_metric < self.best_metric:
            self.best_metric = current_metric
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            
        return self.patience_counter >= self.config.training.early_stopping_patience
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Save model checkpoint"""
        checkpoint_path = os.path.join(
            self.config.output_dir,
            f"checkpoint_epoch_{epoch + 1}.pt"
        )
        
        save_model(
            model=self.model,
            save_path=checkpoint_path,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch,
            metrics=metrics
        )
        
        # Save best model
        if metrics.get('wer', float('inf')) <= self.best_metric:
            best_path = os.path.join(self.config.output_dir, "best_model.pt")
            save_model(
                model=self.model,
                save_path=best_path,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                metrics=metrics
            )
    
    def evaluate(self, test_dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on test set
        
        Args:
            test_dataloader: Test data loader
            
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        
        predictions = []
        references = []
        
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Evaluation"):
                batch = self._move_batch_to_device(batch)
                
                # Generate predictions
                generated_outputs = self.model.generate(
                    audio=batch['audio'],
                    audio_lengths=batch['audio_lengths'],
                    max_length=100,
                    do_sample=False
                )
                
                # Decode predictions
                batch_predictions = self.text_processor.batch_decode(
                    generated_outputs['generated_tokens'],
                    skip_special_tokens=True
                )
                
                predictions.extend(batch_predictions)
                references.extend(batch['original_texts'])
        
        # Compute final metrics
        metrics = calculate_accuracy_metrics(predictions, references)
        
        self.logger.info("Evaluation Results:")
        for key, value in metrics.items():
            self.logger.info(f"{key}: {value:.4f}")
        
        return metrics