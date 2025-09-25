# Multimodal ASR

An advanced Automatic Speech Recognition (ASR) model that leverages multimodal approaches to improve speech recognition accuracy. This implementation combines audio features with optional text context and visual information using transformer-based architectures.

## Features

- **Multimodal Architecture**: Combines audio, text context, and visual features
- **Transformer-Based**: Uses state-of-the-art transformer models for encoding and decoding
- **Flexible Fusion**: Supports multiple fusion strategies (concatenation, attention, cross-attention)
- **Extensible Design**: Easy to customize and extend for different use cases
- **Comprehensive Training**: Complete training pipeline with metrics and evaluation
- **Production Ready**: Includes model saving/loading, configuration management, and inference utilities

## Architecture Overview

The multimodal ASR system consists of several key components:

1. **Audio Encoder**: Processes raw audio using mel-spectrograms and transformer layers
2. **Fusion Module**: Combines multimodal inputs using various fusion strategies
3. **Text Decoder**: Generates transcriptions using transformer decoder
4. **Training Pipeline**: Complete training loop with loss functions and metrics

## Installation

### From Source

```bash
git clone https://github.com/NgDMau/multimodal_asr.git
cd multimodal_asr
pip install -r requirements.txt
pip install -e .
```

### Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- torchaudio >= 2.0.0
- transformers >= 4.20.0
- Other dependencies listed in `requirements.txt`

## Quick Start

### Basic Usage

```python
import torch
from multimodal_asr import MultimodalASR
from multimodal_asr.data import AudioProcessor, TextProcessor
from multimodal_asr.utils import create_model_config

# Initialize processors
audio_processor = AudioProcessor(sample_rate=16000, n_mels=80)
text_processor = TextProcessor(vocab_size=1000)

# Build vocabulary (in practice, use your dataset)
sample_texts = ["hello world", "speech recognition", "multimodal approach"]
text_processor.build_vocabulary(sample_texts)

# Create model
config = create_model_config(
    vocab_size=text_processor.vocabulary_size,
    use_text_context=False,
    fusion_type="attention"
)
model = MultimodalASR(**config)

# Inference
model.eval()
with torch.no_grad():
    # Load your audio file here
    audio = torch.randn(1, 48000)  # 3 seconds at 16kHz
    
    outputs = model.generate(
        audio=audio,
        max_length=100,
        do_sample=False
    )
    
    # Decode predictions
    transcription = text_processor.batch_decode(
        outputs['generated_tokens'],
        skip_special_tokens=True
    )[0]
    
    print(f"Transcription: {transcription}")
```

### Training Example

```python
from multimodal_asr.training import Trainer, MultimodalLoss
from multimodal_asr.data import MultimodalASRDataset, create_data_loader

# Prepare your dataset
train_data = [
    {"audio_path": "path/to/audio1.wav", "text": "hello world"},
    {"audio_path": "path/to/audio2.wav", "text": "speech recognition"},
    # ... more samples
]

# Create dataset and dataloader
train_dataset = MultimodalASRDataset(
    data_list=train_data,
    audio_processor=audio_processor,
    text_processor=text_processor
)
train_loader = create_data_loader(train_dataset, batch_size=8)

# Initialize training components
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss_fn = MultimodalLoss(vocab_size=text_processor.vocabulary_size)

# Create trainer
trainer = Trainer(
    model=model,
    train_dataloader=train_loader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    text_processor=text_processor,
    config=config,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

# Start training
trainer.train()
```

## Multimodal Features

### Text Context

```python
# Enable text context in model configuration
config = create_model_config(
    vocab_size=text_processor.vocabulary_size,
    use_text_context=True,
    fusion_type="cross_attention"
)

model = MultimodalASR(**config)

# Use text context during inference
context_text = "This is about speech recognition"
context_tokens = text_processor.encode(context_text, add_special_tokens=False)
context_tokens = torch.tensor([context_tokens])

outputs = model.generate(
    audio=audio,
    text_context=context_tokens,
    max_length=100
)
```

### Visual Features

```python
# Enable visual features
config = create_model_config(
    vocab_size=text_processor.vocabulary_size,
    use_visual_features=True,
    visual_dim=512,
    fusion_type="attention"
)

model = MultimodalASR(**config)

# Use visual features during inference
visual_features = torch.randn(1, 100, 512)  # [batch, time, features]
outputs = model.generate(
    audio=audio,
    visual_features=visual_features,
    max_length=100
)
```

## Model Configuration

The model supports various configuration options:

```python
from multimodal_asr.utils import Config

config = Config()

# Audio settings
config.audio.sample_rate = 16000
config.audio.n_mels = 80

# Model architecture
config.model.vocab_size = 10000
config.model.audio_hidden_dim = 512
config.model.text_hidden_dim = 512
config.model.fusion_type = "attention"  # "concat", "attention", "cross_attention"
config.model.use_text_context = False
config.model.use_visual_features = False

# Training settings
config.training.batch_size = 8
config.training.learning_rate = 1e-4
config.training.num_epochs = 100
```

## Examples

Complete examples are available in the `examples/` directory:

- `examples/basic_usage.py`: Basic model usage and inference
- `examples/train_example.py`: Complete training example with dummy data

Run examples:

```bash
python examples/basic_usage.py
python examples/train_example.py
```

## Evaluation Metrics

The package includes comprehensive evaluation metrics:

- **Word Error Rate (WER)**: Primary metric for ASR evaluation
- **Character Error Rate (CER)**: Character-level accuracy
- **BLEU Scores**: N-gram based similarity metrics
- **Exact Match**: Percentage of perfectly transcribed samples

```python
from multimodal_asr.utils.metrics import calculate_accuracy_metrics

predictions = ["hello world", "speech recognition"]
references = ["hello world", "speech recognition system"]

metrics = calculate_accuracy_metrics(predictions, references)
print(f"WER: {metrics['wer']:.4f}")
print(f"CER: {metrics['cer']:.4f}")
```

## Model Architecture Details

### Audio Encoder
- Mel-spectrogram feature extraction
- Convolutional layers for local feature extraction
- Transformer encoder for sequence modeling
- Positional encoding for temporal information

### Fusion Module
- Multiple fusion strategies supported
- Learnable modality weights
- Cross-attention between modalities
- Adaptive feature combination

### Text Decoder
- Transformer decoder architecture
- Autoregressive text generation
- Support for beam search and sampling
- Special token handling

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{multimodal_asr,
  title={Multimodal ASR: A multimodal approach to Automatic Speech Recognition},
  author={NgDMau},
  year={2024},
  url={https://github.com/NgDMau/multimodal_asr}
}
```

## Acknowledgments

- Built with PyTorch and Transformers
- Inspired by recent advances in multimodal learning
- Thanks to the open-source community for the foundational libraries
