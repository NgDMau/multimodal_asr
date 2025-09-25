"""
Training example for multimodal ASR
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json
from typing import List, Dict, Any

from multimodal_asr import MultimodalASR
from multimodal_asr.data import AudioProcessor, TextProcessor, MultimodalASRDataset, create_data_loader
from multimodal_asr.training import Trainer, MultimodalLoss
from multimodal_asr.utils import Config, create_model_config, save_config


def create_dummy_dataset(num_samples: int = 100) -> List[Dict[str, Any]]:
    """
    Create a dummy dataset for demonstration
    In practice, this would load real audio files and transcriptions
    """
    import numpy as np
    import soundfile as sf
    
    dataset = []
    
    # Create dummy directory
    dummy_audio_dir = "/tmp/dummy_audio"
    os.makedirs(dummy_audio_dir, exist_ok=True)
    
    # Sample texts for generation
    sample_texts = [
        "hello world this is a test",
        "speech recognition using deep learning",
        "artificial intelligence and machine learning",
        "multimodal approach for better accuracy",
        "transformer models are very powerful",
        "automatic speech recognition systems",
        "natural language processing techniques",
        "audio signal processing methods",
        "sequence to sequence learning",
        "attention mechanisms in neural networks"
    ]
    
    for i in range(num_samples):
        # Generate random audio (3-5 seconds)
        duration = np.random.uniform(3.0, 5.0)
        sample_rate = 16000
        audio_length = int(duration * sample_rate)
        
        # Create dummy audio (normally distributed noise)
        dummy_audio = np.random.randn(audio_length).astype(np.float32)
        dummy_audio *= 0.1  # Reduce amplitude
        
        # Save dummy audio file
        audio_filename = f"dummy_audio_{i:04d}.wav"
        audio_path = os.path.join(dummy_audio_dir, audio_filename)
        sf.write(audio_path, dummy_audio, sample_rate)
        
        # Select random text
        text = sample_texts[i % len(sample_texts)]
        
        # Add some variation
        if np.random.random() < 0.3:
            # Add extra words sometimes
            extra_words = ["very", "quite", "really", "somewhat", "extremely"]
            extra = np.random.choice(extra_words)
            words = text.split()
            insert_pos = np.random.randint(0, len(words))
            words.insert(insert_pos, extra)
            text = " ".join(words)
        
        dataset.append({
            "audio_path": audio_path,
            "text": text
        })
    
    return dataset


def main():
    """
    Main training example
    """
    print("Multimodal ASR - Training Example")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create configuration
    print("\n1. Creating configuration...")
    config = Config()
    
    # Update configuration for this example
    config.model.vocab_size = 1000
    config.training.batch_size = 4
    config.training.num_epochs = 5
    config.training.learning_rate = 1e-4
    config.training.eval_steps = 50
    config.training.save_steps = 100
    config.training.logging_steps = 10
    config.data.max_audio_length = 80000  # 5 seconds at 16kHz
    config.data.max_text_length = 100
    
    print(f"   Model vocab size: {config.model.vocab_size}")
    print(f"   Batch size: {config.training.batch_size}")
    print(f"   Number of epochs: {config.training.num_epochs}")
    
    # Create dummy dataset
    print("\n2. Creating dummy dataset...")
    train_data = create_dummy_dataset(80)  # 80 samples for training
    val_data = create_dummy_dataset(20)    # 20 samples for validation
    
    print(f"   Training samples: {len(train_data)}")
    print(f"   Validation samples: {len(val_data)}")
    
    # Initialize processors
    print("\n3. Initializing processors...")
    audio_processor = AudioProcessor(
        sample_rate=config.audio.sample_rate,
        n_mels=config.audio.n_mels,
        normalize=config.audio.normalize,
        augment=config.audio.augment
    )
    
    text_processor = TextProcessor(
        vocab_size=config.model.vocab_size,
        min_freq=1,  # Low frequency for small dataset
        lowercase=True,
        remove_punctuation=True
    )
    
    # Build vocabulary from training data
    train_texts = [sample['text'] for sample in train_data]
    text_processor.build_vocabulary(train_texts)
    
    print(f"   Built vocabulary with {text_processor.vocabulary_size} tokens")
    
    # Create datasets
    print("\n4. Creating datasets...")
    train_dataset = MultimodalASRDataset(
        data_list=train_data,
        audio_processor=audio_processor,
        text_processor=text_processor,
        max_audio_length=config.data.max_audio_length,
        max_text_length=config.data.max_text_length
    )
    
    val_dataset = MultimodalASRDataset(
        data_list=val_data,
        audio_processor=audio_processor,
        text_processor=text_processor,
        max_audio_length=config.data.max_audio_length,
        max_text_length=config.data.max_text_length
    )
    
    # Create data loaders
    train_loader = create_data_loader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=2  # Reduced for example
    )
    
    val_loader = create_data_loader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    
    # Initialize model
    print("\n5. Initializing model...")
    model_config = create_model_config(
        vocab_size=text_processor.vocabulary_size,
        sample_rate=config.audio.sample_rate,
        use_text_context=config.model.use_text_context,
        use_visual_features=config.model.use_visual_features,
        fusion_type=config.model.fusion_type
    )
    
    model = MultimodalASR(**model_config)
    model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Initialize optimizer and scheduler
    print("\n6. Setting up optimizer and scheduler...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        betas=(config.training.beta1, config.training.beta2)
    )
    
    # Calculate total steps for scheduler
    total_steps = len(train_loader) * config.training.num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=total_steps,
        eta_min=config.training.learning_rate * 0.01
    )
    
    # Initialize loss function
    loss_fn = MultimodalLoss(
        vocab_size=text_processor.vocabulary_size,
        pad_token_id=text_processor.pad_token_id,
        label_smoothing=config.training.label_smoothing
    )
    
    # Initialize trainer
    print("\n7. Initializing trainer...")
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        text_processor=text_processor,
        config=config,
        device=device
    )
    
    # Start training
    print("\n8. Starting training...")
    print("-" * 50)
    
    try:
        training_history = trainer.train()
        
        print("-" * 50)
        print("Training completed successfully!")
        
        # Print training summary
        print(f"\nTraining Summary:")
        print(f"Final training loss: {training_history['train_loss'][-1]:.4f}")
        if training_history['val_loss']:
            print(f"Final validation loss: {training_history['val_loss'][-1]:.4f}")
        if training_history['val_wer']:
            print(f"Final validation WER: {training_history['val_wer'][-1]:.4f}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    # Demonstrate inference
    print("\n9. Testing inference...")
    model.eval()
    
    # Get a sample batch
    sample_batch = next(iter(val_loader))
    
    # Move to device
    for key in sample_batch:
        if isinstance(sample_batch[key], torch.Tensor):
            sample_batch[key] = sample_batch[key].to(device)
    
    with torch.no_grad():
        # Generate predictions
        outputs = model.generate(
            audio=sample_batch['audio'][:2],  # Just first 2 samples
            audio_lengths=sample_batch['audio_lengths'][:2],
            max_length=50,
            do_sample=False
        )
        
        # Decode predictions
        predictions = text_processor.batch_decode(
            outputs['generated_tokens'],
            skip_special_tokens=True
        )
        
        print("Sample predictions:")
        for i, pred in enumerate(predictions):
            ref = sample_batch['original_texts'][i]
            print(f"   Prediction {i+1}: '{pred}'")
            print(f"   Reference {i+1}:  '{ref}'")
    
    print("\n" + "=" * 50)
    print("Training example completed!")
    print(f"Model checkpoints saved in: {config.output_dir}")
    
    # Save final configuration
    config_path = os.path.join(config.output_dir, "config.json")
    config.save(config_path)
    print(f"Configuration saved to: {config_path}")


if __name__ == "__main__":
    main()