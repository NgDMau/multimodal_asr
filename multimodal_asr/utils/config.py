"""
Configuration management for multimodal ASR
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import json
import os


@dataclass
class AudioConfig:
    """Audio processing configuration"""
    sample_rate: int = 16000
    n_mels: int = 80
    n_fft: int = 512
    hop_length: int = 160
    win_length: int = 400
    normalize: bool = True
    augment: bool = False


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    vocab_size: int = 10000
    
    # Audio encoder
    audio_input_dim: int = 80
    audio_hidden_dim: int = 512
    audio_num_layers: int = 6
    audio_num_heads: int = 8
    
    # Text decoder
    text_hidden_dim: int = 512
    text_num_layers: int = 6
    text_num_heads: int = 8
    max_text_length: int = 512
    
    # Fusion
    fusion_type: str = "attention"  # "concat", "attention", "cross_attention"
    use_text_context: bool = False
    use_visual_features: bool = False
    text_context_dim: int = 512
    visual_dim: Optional[int] = None
    
    # General
    dropout: float = 0.1
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Basic training parameters
    batch_size: int = 8
    learning_rate: float = 1e-4
    num_epochs: int = 100
    warmup_steps: int = 1000
    gradient_clip_norm: float = 1.0
    
    # Optimizer
    optimizer: str = "adamw"  # "adam", "adamw", "sgd"
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    
    # Scheduler
    scheduler: str = "cosine"  # "linear", "cosine", "constant"
    scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Loss
    label_smoothing: float = 0.1
    
    # Validation and saving
    eval_steps: int = 1000
    save_steps: int = 5000
    logging_steps: int = 100
    max_checkpoints: int = 3
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_metric: str = "wer"  # "wer", "loss"
    
    # Mixed precision
    use_fp16: bool = False
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class DataConfig:
    """Data configuration"""
    # Dataset paths
    train_data: str = ""
    val_data: str = ""
    test_data: str = ""
    
    # Audio processing
    max_audio_length: Optional[int] = None
    max_text_length: Optional[int] = 512
    
    # Vocabulary
    vocab_file: Optional[str] = None
    build_vocab: bool = True
    min_freq: int = 2
    
    # Data loading
    shuffle_train: bool = True
    prefetch_factor: int = 2


@dataclass
class Config:
    """Main configuration class"""
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # Experiment settings
    experiment_name: str = "multimodal_asr"
    output_dir: str = "./outputs"
    log_dir: str = "./logs"
    seed: int = 42
    
    def save(self, config_path: str) -> None:
        """Save configuration to JSON file"""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        config_dict = self.to_dict()
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        print(f"Configuration saved to {config_path}")
    
    @classmethod
    def load(cls, config_path: str) -> 'Config':
        """Load configuration from JSON file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'audio': {
                'sample_rate': self.audio.sample_rate,
                'n_mels': self.audio.n_mels,
                'n_fft': self.audio.n_fft,
                'hop_length': self.audio.hop_length,
                'win_length': self.audio.win_length,
                'normalize': self.audio.normalize,
                'augment': self.audio.augment
            },
            'model': {
                'vocab_size': self.model.vocab_size,
                'audio_input_dim': self.model.audio_input_dim,
                'audio_hidden_dim': self.model.audio_hidden_dim,
                'audio_num_layers': self.model.audio_num_layers,
                'audio_num_heads': self.model.audio_num_heads,
                'text_hidden_dim': self.model.text_hidden_dim,
                'text_num_layers': self.model.text_num_layers,
                'text_num_heads': self.model.text_num_heads,
                'max_text_length': self.model.max_text_length,
                'fusion_type': self.model.fusion_type,
                'use_text_context': self.model.use_text_context,
                'use_visual_features': self.model.use_visual_features,
                'text_context_dim': self.model.text_context_dim,
                'visual_dim': self.model.visual_dim,
                'dropout': self.model.dropout,
                'pad_token_id': self.model.pad_token_id,
                'bos_token_id': self.model.bos_token_id,
                'eos_token_id': self.model.eos_token_id
            },
            'training': {
                'batch_size': self.training.batch_size,
                'learning_rate': self.training.learning_rate,
                'num_epochs': self.training.num_epochs,
                'warmup_steps': self.training.warmup_steps,
                'gradient_clip_norm': self.training.gradient_clip_norm,
                'optimizer': self.training.optimizer,
                'weight_decay': self.training.weight_decay,
                'beta1': self.training.beta1,
                'beta2': self.training.beta2,
                'scheduler': self.training.scheduler,
                'scheduler_kwargs': self.training.scheduler_kwargs,
                'label_smoothing': self.training.label_smoothing,
                'eval_steps': self.training.eval_steps,
                'save_steps': self.training.save_steps,
                'logging_steps': self.training.logging_steps,
                'max_checkpoints': self.training.max_checkpoints,
                'early_stopping_patience': self.training.early_stopping_patience,
                'early_stopping_metric': self.training.early_stopping_metric,
                'use_fp16': self.training.use_fp16,
                'num_workers': self.training.num_workers,
                'pin_memory': self.training.pin_memory
            },
            'data': {
                'train_data': self.data.train_data,
                'val_data': self.data.val_data,
                'test_data': self.data.test_data,
                'max_audio_length': self.data.max_audio_length,
                'max_text_length': self.data.max_text_length,
                'vocab_file': self.data.vocab_file,
                'build_vocab': self.data.build_vocab,
                'min_freq': self.data.min_freq,
                'shuffle_train': self.data.shuffle_train,
                'prefetch_factor': self.data.prefetch_factor
            },
            'experiment_name': self.experiment_name,
            'output_dir': self.output_dir,
            'log_dir': self.log_dir,
            'seed': self.seed
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create configuration from dictionary"""
        config = cls()
        
        # Audio config
        if 'audio' in config_dict:
            audio_dict = config_dict['audio']
            config.audio = AudioConfig(**audio_dict)
        
        # Model config
        if 'model' in config_dict:
            model_dict = config_dict['model']
            config.model = ModelConfig(**model_dict)
        
        # Training config
        if 'training' in config_dict:
            training_dict = config_dict['training']
            config.training = TrainingConfig(**training_dict)
        
        # Data config
        if 'data' in config_dict:
            data_dict = config_dict['data']
            config.data = DataConfig(**data_dict)
        
        # Experiment settings
        config.experiment_name = config_dict.get('experiment_name', config.experiment_name)
        config.output_dir = config_dict.get('output_dir', config.output_dir)
        config.log_dir = config_dict.get('log_dir', config.log_dir)
        config.seed = config_dict.get('seed', config.seed)
        
        return config
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values"""
        for key, value in updates.items():
            if '.' in key:
                # Handle nested updates like 'model.vocab_size'
                parts = key.split('.')
                obj = self
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
            else:
                setattr(self, key, value)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of warnings/errors"""
        warnings = []
        
        # Check required data paths
        if not self.data.train_data:
            warnings.append("No training data path specified")
        
        # Check vocabulary consistency
        if self.model.vocab_size <= 0:
            warnings.append("Vocabulary size must be positive")
        
        # Check model dimensions
        if self.model.audio_hidden_dim != self.model.text_hidden_dim:
            warnings.append("Audio and text hidden dimensions should match for optimal fusion")
        
        # Check training parameters
        if self.training.learning_rate <= 0:
            warnings.append("Learning rate must be positive")
        
        if self.training.batch_size <= 0:
            warnings.append("Batch size must be positive")
        
        return warnings
    
    def create_directories(self) -> None:
        """Create necessary directories"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)


def create_default_config() -> Config:
    """Create a default configuration"""
    return Config()


def load_config_from_args(args) -> Config:
    """Load configuration from command line arguments (placeholder)"""
    # This would typically parse command line arguments
    # For now, return default config
    config = create_default_config()
    
    # Override with any provided arguments
    if hasattr(args, 'config_file') and args.config_file:
        config = Config.load(args.config_file)
    
    return config