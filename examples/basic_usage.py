"""
Basic usage example for multimodal ASR
"""

import torch
import numpy as np
from multimodal_asr import MultimodalASR
from multimodal_asr.data import AudioProcessor, TextProcessor
from multimodal_asr.utils import create_model_config


def main():
    """
    Demonstrate basic usage of the multimodal ASR model
    """
    print("Multimodal ASR - Basic Usage Example")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize text processor
    print("\n1. Initializing text processor...")
    text_processor = TextProcessor(vocab_size=1000, lowercase=True)
    
    # Build a simple vocabulary from sample texts
    sample_texts = [
        "hello world this is a test",
        "speech recognition using multimodal approach",
        "artificial intelligence and machine learning",
        "deep learning for audio processing",
        "transformer models for sequence to sequence tasks"
    ]
    text_processor.build_vocabulary(sample_texts)
    
    # Initialize audio processor
    print("2. Initializing audio processor...")
    audio_processor = AudioProcessor(
        sample_rate=16000,
        n_mels=80,
        normalize=True,
        augment=False
    )
    
    # Create model configuration
    print("3. Creating model configuration...")
    config = create_model_config(
        vocab_size=text_processor.vocabulary_size,
        sample_rate=16000,
        use_text_context=False,
        use_visual_features=False,
        fusion_type="attention"
    )
    
    # Initialize model
    print("4. Initializing multimodal ASR model...")
    model = MultimodalASR(**config)
    model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Create dummy audio data (simulating 3 seconds of audio)
    print("\n5. Creating dummy audio data...")
    batch_size = 2
    audio_length = 16000 * 3  # 3 seconds at 16kHz
    
    # Generate random audio (in practice, this would be real audio)
    dummy_audio = torch.randn(batch_size, audio_length)
    audio_lengths = torch.tensor([audio_length, audio_length // 2])  # Different lengths
    
    # Create target texts
    target_texts = [
        "hello world this is a test",
        "speech recognition works"
    ]
    
    # Encode target texts
    print("6. Encoding target texts...")
    encoded_targets = []
    target_lengths = []
    
    for text in target_texts:
        tokens = text_processor.encode(text, add_special_tokens=True)
        encoded_targets.append(tokens)
        target_lengths.append(len(tokens))
    
    # Pad sequences to same length
    max_length = max(len(tokens) for tokens in encoded_targets)
    padded_targets = []
    
    for tokens in encoded_targets:
        padded = tokens + [text_processor.pad_token_id] * (max_length - len(tokens))
        padded_targets.append(padded)
    
    target_tokens = torch.tensor(padded_targets, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    
    # Move to device
    dummy_audio = dummy_audio.to(device)
    audio_lengths = audio_lengths.to(device)
    target_tokens = target_tokens.to(device)
    target_lengths = target_lengths.to(device)
    
    # Forward pass (training mode)
    print("\n7. Running forward pass (training mode)...")
    model.train()
    
    with torch.no_grad():  # No gradients for demo
        outputs = model(
            audio=dummy_audio,
            audio_lengths=audio_lengths,
            target_tokens=target_tokens,
            target_lengths=target_lengths
        )
        
        logits = outputs['logits']
        print(f"   Output logits shape: {logits.shape}")
        print(f"   Audio features shape: {outputs['audio_features'].shape}")
    
    # Inference (generation mode)
    print("\n8. Running inference (generation mode)...")
    model.eval()
    
    with torch.no_grad():
        generated_outputs = model.generate(
            audio=dummy_audio,
            audio_lengths=audio_lengths,
            max_length=50,
            do_sample=False,  # Greedy decoding
            temperature=1.0
        )
        
        generated_tokens = generated_outputs['generated_tokens']
        print(f"   Generated tokens shape: {generated_tokens.shape}")
        
        # Decode generated tokens back to text
        generated_texts = text_processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )
        
        print("\n   Generated transcriptions:")
        for i, text in enumerate(generated_texts):
            print(f"   Sample {i+1}: '{text}'")
            print(f"   Target {i+1}:  '{target_texts[i]}'")
    
    # Demonstrate text context usage
    print("\n9. Demonstrating text context usage...")
    
    # Create model with text context
    config_with_context = create_model_config(
        vocab_size=text_processor.vocabulary_size,
        sample_rate=16000,
        use_text_context=True,
        use_visual_features=False,
        fusion_type="cross_attention"
    )
    
    model_with_context = MultimodalASR(**config_with_context)
    model_with_context.to(device)
    model_with_context.eval()
    
    # Create dummy text context
    context_texts = [
        "this is about speech recognition",
        "talking about artificial intelligence"
    ]
    
    encoded_contexts = []
    context_lengths = []
    
    for text in context_texts:
        tokens = text_processor.encode(text, add_special_tokens=False)
        encoded_contexts.append(tokens)
        context_lengths.append(len(tokens))
    
    # Pad contexts
    max_context_length = max(len(tokens) for tokens in encoded_contexts)
    padded_contexts = []
    
    for tokens in encoded_contexts:
        padded = tokens + [text_processor.pad_token_id] * (max_context_length - len(tokens))
        padded_contexts.append(padded)
    
    context_tokens = torch.tensor(padded_contexts, dtype=torch.long).to(device)
    context_lengths = torch.tensor(context_lengths, dtype=torch.long).to(device)
    
    # Generate with text context
    with torch.no_grad():
        context_outputs = model_with_context.generate(
            audio=dummy_audio,
            audio_lengths=audio_lengths,
            text_context=context_tokens,
            text_context_lengths=context_lengths,
            max_length=50,
            do_sample=False
        )
        
        context_generated_texts = text_processor.batch_decode(
            context_outputs['generated_tokens'],
            skip_special_tokens=True
        )
        
        print("   Generated transcriptions with text context:")
        for i, text in enumerate(context_generated_texts):
            print(f"   Sample {i+1}: '{text}'")
            print(f"   Context {i+1}: '{context_texts[i]}'")
    
    print("\n" + "=" * 50)
    print("Basic usage example completed!")
    print("Note: This used dummy data. In practice, you would:")
    print("1. Load real audio files")
    print("2. Train the model on your dataset")
    print("3. Use trained model for inference")


if __name__ == "__main__":
    main()