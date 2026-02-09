#!/usr/bin/env python3
"""
Demo script for Telugu Agricultural SLM
Quick demonstration of the model capabilities
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from model import TeluguAgriSLM
from tokenizer import TeluguAgriTokenizer


def demo_tokenizer():
    """Demonstrate tokenizer functionality"""
    print("=" * 60)
    print("TOKENIZER DEMO")
    print("=" * 60)
    
    # Create and train a small tokenizer for demo
    from tokenizer import create_sample_training_data
    
    print("\n1. Creating sample training data...")
    data_file = "data/demo_tokenizer_train.txt"
    create_sample_training_data(data_file, num_samples=1000)
    
    print("\n2. Training tokenizer...")
    tokenizer = TeluguAgriTokenizer(vocab_size=800)  # Smaller vocab for demo
    tokenizer.train_sentencepiece(data_file)
    
    print("\n3. Testing tokenizer:")
    
    test_sentences = [
        "వరి పంటకు ఎంత నీరు అవసరం?",
        "మొక్కజొన్న ఎప్పుడు వేయాలి?",
        "పత్తి పంటలో తెగులు నివారణ చర్యలు",
        "సేంద్రీయ వ్యవసాయం ప్రయోజనాలు",
    ]
    
    for text in test_sentences:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        print(f"\n  Input:  {text}")
        print(f"  Tokens: {len(encoded)}")
        print(f"  IDs:    {encoded[:5]}...{encoded[-5:]}")
        print(f"  Output: {decoded}")
    
    print(f"\n✅ Tokenizer demo complete!")
    return tokenizer


def demo_model(tokenizer):
    """Demonstrate model functionality"""
    print("\n" + "=" * 60)
    print("MODEL DEMO")
    print("=" * 60)
    
    print("\n1. Creating model...")
    model = TeluguAgriSLM(
        vocab_size=tokenizer.sp_model.get_piece_size(),
        d_model=128,      # Small for demo
        n_layers=2,       # Small for demo
        n_heads=4,
        d_ff=512,
        max_seq_len=256,
    )
    
    print(f"\n2. Model has {model.count_parameters()/1e6:.2f}M parameters")
    
    print("\n3. Testing forward pass...")
    import torch
    
    # Test forward pass
    test_text = "వరి పంటకు నీరు అవసరం"
    input_ids = tokenizer.encode(test_text, add_special_tokens=True)
    input_tensor = torch.tensor([input_ids], dtype=torch.long)
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor, labels=input_tensor)
    
    print(f"  Input: {test_text}")
    print(f"  Loss: {outputs['loss'].item():.4f}")
    print(f"  Logits shape: {outputs['logits'].shape}")
    
    print("\n4. Testing generation (random weights - output will be gibberish)...")
    prompt = "వరి పంట"
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_tensor = torch.tensor([input_ids], dtype=torch.long)
    
    with torch.no_grad():
        generated = model.generate(
            input_tensor,
            max_new_tokens=20,
            temperature=0.7,
            top_k=50,
        )
    
    output_text = tokenizer.decode(generated[0].tolist())
    print(f"  Prompt:  {prompt}")
    print(f"  Generated: {output_text}")
    
    print(f"\n✅ Model demo complete!")
    return model


def demo_data_pipeline():
    """Demonstrate data pipeline"""
    print("\n" + "=" * 60)
    print("DATA PIPELINE DEMO")
    print("=" * 60)
    
    from data_pipeline import TeluguAgriDataCollector
    
    print("\n1. Generating synthetic data...")
    collector = TeluguAgriDataCollector(output_dir="data/demo_raw")
    
    # Generate smaller dataset for demo
    qa_pairs = collector.generate_qa_pairs(10)
    documents = collector.generate_agricultural_text(10)
    conversations = collector.generate_conversations(5)
    
    print(f"  Generated {len(qa_pairs)} Q&A pairs")
    print(f"  Generated {len(documents)} documents")
    print(f"  Generated {len(conversations)} conversations")
    
    print("\n2. Sample Q&A:")
    for i, qa in enumerate(qa_pairs[:3], 1):
        print(f"\n  Q{i}: {qa['question']}")
        print(f"  A{i}: {qa['answer'][:100]}...")
    
    print("\n3. Sample Document:")
    print(f"  {documents[0][:200]}...")
    
    print(f"\n✅ Data pipeline demo complete!")


def demo_full_workflow():
    """Demonstrate the full workflow"""
    print("\n" + "=" * 60)
    print("FULL WORKFLOW DEMO")
    print("=" * 60)
    
    import torch
    
    # Step 1: Create tokenizer
    print("\n[1/4] Training tokenizer...")
    from tokenizer import create_sample_training_data
    data_file = "data/demo_workflow.txt"
    create_sample_training_data(data_file, num_samples=2000)
    
    tokenizer = TeluguAgriTokenizer(vocab_size=800)  # Smaller for demo
    tokenizer.train_sentencepiece(data_file)
    
    # Step 2: Create model
    print("[2/4] Creating model...")
    model = TeluguAgriSLM(
        vocab_size=tokenizer.sp_model.get_piece_size(),
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_ff=256,
        max_seq_len=128,
    )
    
    # Step 3: Generate some training data
    print("[3/4] Preparing training data...")
    from data_pipeline import TeluguAgriDataCollector
    collector = TeluguAgriDataCollector("data/demo_workflow")
    
    documents = collector.generate_agricultural_text(50)
    corpus_file = "data/demo_workflow/corpus.txt"
    with open(corpus_file, 'w', encoding='utf-8') as f:
        for doc in documents:
            f.write(doc + '\n\n')
    
    # Step 4: Mock training (just a few steps)
    print("[4/4] Mock training (5 steps)...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Create simple dataset
    texts = documents[:10]
    for step in range(5):
        text = texts[step % len(texts)]
        input_ids = tokenizer.encode(text, add_special_tokens=True)
        
        # Truncate
        input_ids = input_ids[:128]
        input_tensor = torch.tensor([input_ids], dtype=torch.long)
        
        optimizer.zero_grad()
        outputs = model(input_tensor, labels=input_tensor)
        loss = outputs['loss']
        loss.backward()
        optimizer.step()
        
        print(f"  Step {step+1}: Loss = {loss.item():.4f}")
    
    # Test generation after "training"
    print("\n5. Testing generation after training...")
    model.eval()
    
    prompt = "వరి పంట"
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_tensor = torch.tensor([input_ids], dtype=torch.long)
    
    with torch.no_grad():
        generated = model.generate(
            input_tensor,
            max_new_tokens=15,
            temperature=0.7,
        )
    
    output_text = tokenizer.decode(generated[0].tolist())
    print(f"  Prompt:  {prompt}")
    print(f"  Generated: {output_text}")
    
    print(f"\n✅ Full workflow demo complete!")
    print("\nNote: This is a minimal demo with tiny models and few steps.")
    print("For a real model, run: python pipeline.py all")


def main():
    """Run all demos"""
    print("\n" + "=" * 70)
    print("TELUGU AGRICULTURAL SLM - DEMO")
    print("=" * 70)
    print("\nThis script demonstrates the key components of the SLM.")
    print("Each section shows a different part of the system.")
    
    try:
        # Run demos
        tokenizer = demo_tokenizer()
        demo_model(tokenizer)
        demo_data_pipeline()
        demo_full_workflow()
        
        print("\n" + "=" * 70)
        print("🎉 ALL DEMOS COMPLETE!")
        print("=" * 70)
        print("\nNext steps:")
        print("  - Train full model: python pipeline.py all")
        print("  - Run inference: python pipeline.py inference")
        print("  - Start API: python src/inference.py --mode api")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
