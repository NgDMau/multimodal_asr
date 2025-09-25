"""
Model utilities for saving/loading and configuration
"""

import torch
import os
import json
from typing import Dict, Any, Optional
from ..models.multimodal_asr import MultimodalASR


def save_model(
    model: MultimodalASR,
    save_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    metrics: Optional[Dict[str, float]] = None
) -> None:
    """
    Save model checkpoint
    
    Args:
        model: MultimodalASR model to save
        save_path: Path to save checkpoint
        optimizer: Optional optimizer state
        scheduler: Optional scheduler state
        epoch: Current epoch number
        metrics: Optional metrics dictionary
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'vocab_size': model.vocab_size,
            'use_text_context': model.use_text_context,
            'use_visual_features': model.use_visual_features,
            'pad_token_id': model.pad_token_id,
            'bos_token_id': model.bos_token_id,
            'eos_token_id': model.eos_token_id
        },
        'epoch': epoch,
        'metrics': metrics or {}
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, save_path)
    print(f"Model checkpoint saved to {save_path}")


def load_model(
    checkpoint_path: str,
    model_class: type = MultimodalASR,
    map_location: Optional[str] = None,
    **model_kwargs
) -> Dict[str, Any]:
    """
    Load model checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        model_class: Model class to instantiate
        map_location: Device to map tensors to
        **model_kwargs: Additional model initialization arguments
        
    Returns:
        Dictionary containing model and metadata
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    
    # Get model configuration
    model_config = checkpoint.get('model_config', {})
    
    # Merge with provided kwargs
    model_config.update(model_kwargs)
    
    # Instantiate model
    model = model_class(**model_config)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    result = {
        'model': model,
        'epoch': checkpoint.get('epoch', 0),
        'metrics': checkpoint.get('metrics', {}),
        'model_config': model_config
    }
    
    # Add optimizer and scheduler states if available
    if 'optimizer_state_dict' in checkpoint:
        result['optimizer_state_dict'] = checkpoint['optimizer_state_dict']
    
    if 'scheduler_state_dict' in checkpoint:
        result['scheduler_state_dict'] = checkpoint['scheduler_state_dict']
    
    print(f"Model checkpoint loaded from {checkpoint_path}")
    return result


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }


def get_model_size_mb(model: torch.nn.Module) -> float:
    """
    Get model size in megabytes
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def freeze_module(module: torch.nn.Module) -> None:
    """
    Freeze all parameters in a module
    
    Args:
        module: Module to freeze
    """
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module: torch.nn.Module) -> None:
    """
    Unfreeze all parameters in a module
    
    Args:
        module: Module to unfreeze
    """
    for param in module.parameters():
        param.requires_grad = True


def initialize_weights(model: torch.nn.Module, init_type: str = 'xavier') -> None:
    """
    Initialize model weights
    
    Args:
        model: Model to initialize
        init_type: Initialization type ('xavier', 'kaiming', 'normal')
    """
    for name, param in model.named_parameters():
        if 'weight' in name:
            if init_type == 'xavier':
                if len(param.shape) >= 2:
                    torch.nn.init.xavier_uniform_(param)
            elif init_type == 'kaiming':
                if len(param.shape) >= 2:
                    torch.nn.init.kaiming_uniform_(param)
            elif init_type == 'normal':
                torch.nn.init.normal_(param, mean=0, std=0.02)
        elif 'bias' in name:
            torch.nn.init.constant_(param, 0)


def create_model_config(
    vocab_size: int,
    sample_rate: int = 16000,
    use_text_context: bool = False,
    use_visual_features: bool = False,
    fusion_type: str = "attention"
) -> Dict[str, Any]:
    """
    Create default model configuration
    
    Args:
        vocab_size: Vocabulary size
        sample_rate: Audio sample rate
        use_text_context: Whether to use text context
        use_visual_features: Whether to use visual features
        fusion_type: Type of fusion to use
        
    Returns:
        Model configuration dictionary
    """
    config = {
        'vocab_size': vocab_size,
        'sample_rate': sample_rate,
        'use_text_context': use_text_context,
        'use_visual_features': use_visual_features,
        'fusion_type': fusion_type,
        'audio_hidden_dim': 512,
        'text_hidden_dim': 512,
        'audio_num_layers': 6,
        'text_num_layers': 6,
        'audio_num_heads': 8,
        'text_num_heads': 8,
        'dropout': 0.1,
        'n_mels': 80,
        'max_text_length': 512
    }
    
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to JSON file
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"Configuration saved to {config_path}")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print(f"Configuration loaded from {config_path}")
    return config


def print_model_summary(model: torch.nn.Module) -> None:
    """
    Print model summary with parameter counts
    
    Args:
        model: Model to summarize
    """
    print("=" * 80)
    print(f"Model: {model.__class__.__name__}")
    print("=" * 80)
    
    param_counts = count_parameters(model)
    model_size = get_model_size_mb(model)
    
    print(f"Total parameters: {param_counts['total_parameters']:,}")
    print(f"Trainable parameters: {param_counts['trainable_parameters']:,}")
    print(f"Non-trainable parameters: {param_counts['non_trainable_parameters']:,}")
    print(f"Model size: {model_size:.2f} MB")
    print("=" * 80)
    
    # Print module-wise parameter counts
    print("\nModule-wise parameter breakdown:")
    print("-" * 50)
    
    for name, module in model.named_children():
        module_params = count_parameters(module)
        print(f"{name:25s}: {module_params['trainable_parameters']:,}")
    
    print("-" * 50)