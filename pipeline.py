#!/usr/bin/env python3
"""
Main Pipeline for Telugu Agricultural SLM
Orchestrates the entire workflow from data collection to evaluation
"""
import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_pipeline import TeluguAgriDataCollector, DataProcessor
from tokenizer import TeluguAgriTokenizer, create_sample_training_data
from model import TeluguAgriSLM
from train import TeluguAgriDataset, Trainer
from inference import TeluguAgriInference
from evaluate import Evaluator


def setup_directories():
    """Create necessary directories"""
    dirs = [
        "data/raw",
        "data/processed",
        "data/cache",
        "models/tokenizer",
        "models/checkpoints",
        "models/logs",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("Directories created.")


def step1_collect_data():
    """Step 1: Collect and prepare data"""
    print("\n" + "=" * 60)
    print("STEP 1: Data Collection")
    print("=" * 60)
    
    collector = TeluguAgriDataCollector()
    stats = collector.collect_all()
    
    print(f"\nData collection complete!")
    print(f"  Q&A pairs: {stats['qa_pairs']}")
    print(f"  Documents: {stats['documents']}")
    print(f"  Conversations: {stats['conversations']}")
    
    # Process and split data
    processor = DataProcessor()
    processor.split_train_val_test()
    
    return True


def step2_train_tokenizer():
    """Step 2: Train tokenizer"""
    print("\n" + "=" * 60)
    print("STEP 2: Tokenizer Training")
    print("=" * 60)
    
    # Create training data for tokenizer
    data_file = "data/tokenizer_train.txt"
    create_sample_training_data(data_file, num_samples=10000)
    
    # Train tokenizer
    tokenizer = TeluguAgriTokenizer(vocab_size=32000)
    tokenizer.train_sentencepiece(data_file)
    
    # Test tokenizer
    test_text = "వరి పంటకు ఎంత నీరు అవసరం?"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"\nTokenizer test:")
    print(f"  Original: {test_text}")
    print(f"  Encoded: {encoded[:10]}... (len={len(encoded)})")
    print(f"  Decoded: {decoded}")
    
    return True


def step3_train_model(quick_mode: bool = False):
    """Step 3: Train the model"""
    print("\n" + "=" * 60)
    print("STEP 3: Model Training")
    print("=" * 60)
    
    # Load tokenizer
    tokenizer_path = "models/tokenizer/telugu_agri_spm.model"
    if not os.path.exists(tokenizer_path):
        print(f"Tokenizer not found at {tokenizer_path}")
        print("Please run step 2 first.")
        return False
    
    tokenizer = TeluguAgriTokenizer()
    tokenizer.load(tokenizer_path)
    
    # Create model
    model = TeluguAgriSLM(
        vocab_size=tokenizer.sp_model.get_piece_size(),
        d_model=256 if quick_mode else 512,
        n_layers=4 if quick_mode else 8,
        n_heads=4 if quick_mode else 8,
        d_ff=1024 if quick_mode else 2048,
        max_seq_len=512,
    )
    
    # Load datasets
    if not os.path.exists("data/processed/train.txt"):
        print("Training data not found. Please run step 1 first.")
        return False
    
    train_dataset = TeluguAgriDataset(
        "data/processed/train.txt",
        tokenizer,
        max_length=512,
    )
    
    val_dataset = None
    if os.path.exists("data/processed/val.txt"):
        val_dataset = TeluguAgriDataset(
            "data/processed/val.txt",
            tokenizer,
            max_length=512,
        )
    
    # Create trainer with modified config for quick mode
    from config import config
    if quick_mode:
        config.training.num_epochs = 1
        config.training.max_steps = 500
        config.training.batch_size = 4
        config.training.save_steps = 250
        config.training.eval_steps = 250
    
    trainer = Trainer(model, tokenizer, train_dataset, val_dataset)
    
    # Train
    trainer.train()
    
    return True


def step4_run_inference(question: str = None):
    """Step 4: Run inference"""
    print("\n" + "=" * 60)
    print("STEP 4: Inference")
    print("=" * 60)
    
    model_path = "models/checkpoints/final/pytorch_model.bin"
    tokenizer_path = "models/tokenizer/telugu_agri_spm.model"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please run step 3 first.")
        return False
    
    inference = TeluguAgriInference(model_path, tokenizer_path)
    
    if question:
        print(f"\nQuestion: {question}")
        answer = inference.answer_question(question)
        print(f"Answer: {answer}")
    else:
        # Interactive mode
        print("\nEnter your questions (type 'quit' to exit):")
        while True:
            user_input = input("\n🌾 మీ ప్రశ్న: ").strip()
            if user_input.lower() in ['quit', 'exit']:
                break
            if user_input:
                answer = inference.answer_question(user_input)
                print(f"🤖 సమాధానం: {answer}")
    
    return True


def step5_evaluate():
    """Step 5: Evaluate the model"""
    print("\n" + "=" * 60)
    print("STEP 5: Evaluation")
    print("=" * 60)
    
    model_path = "models/checkpoints/final/pytorch_model.bin"
    tokenizer_path = "models/tokenizer/telugu_agri_spm.model"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please run step 3 first.")
        return False
    
    evaluator = Evaluator(model_path, tokenizer_path)
    results = evaluator.run_full_evaluation("models/evaluation_results.json")
    
    return True


def run_all(quick_mode: bool = False):
    """Run all steps in sequence"""
    print("\n" + "=" * 70)
    print("TELUGU AGRICULTURAL SLM - FULL PIPELINE")
    print("=" * 70)
    
    if quick_mode:
        print("\n⚡ QUICK MODE: Using smaller model and fewer training steps")
    
    setup_directories()
    
    # Run all steps
    steps = [
        ("Data Collection", step1_collect_data),
        ("Tokenizer Training", step2_train_tokenizer),
        ("Model Training", lambda: step3_train_model(quick_mode)),
        ("Evaluation", step5_evaluate),
    ]
    
    for name, step_func in steps:
        try:
            success = step_func()
            if not success:
                print(f"\n❌ {name} failed. Stopping pipeline.")
                return False
            print(f"\n✅ {name} complete!")
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n" + "=" * 70)
    print("🎉 ALL STEPS COMPLETE!")
    print("=" * 70)
    print("\nYou can now run inference with:")
    print("  python pipeline.py inference --question 'మీ ప్రశ్న ఇక్కడ'")
    print("\nOr start the API server with:")
    print("  python src/inference.py --mode api")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Telugu Agricultural SLM Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python pipeline.py all

  # Quick mode (smaller model, faster training)
  python pipeline.py all --quick

  # Run individual steps
  python pipeline.py data
  python pipeline.py tokenizer
  python pipeline.py train
  python pipeline.py inference --question "వరి పంట గురించి చెప్పండి"
  python pipeline.py evaluate
        """
    )
    
    parser.add_argument(
        "command",
        choices=["all", "data", "tokenizer", "train", "inference", "evaluate"],
        help="Command to run"
    )
    parser.add_argument("--quick", action="store_true",
                       help="Quick mode with smaller model")
    parser.add_argument("--question", type=str,
                       help="Question for inference mode")
    
    args = parser.parse_args()
    
    # Change to project directory
    os.chdir(Path(__file__).parent)
    
    if args.command == "all":
        run_all(quick_mode=args.quick)
    elif args.command == "data":
        setup_directories()
        step1_collect_data()
    elif args.command == "tokenizer":
        setup_directories()
        step2_train_tokenizer()
    elif args.command == "train":
        step3_train_model(quick_mode=args.quick)
    elif args.command == "inference":
        step4_run_inference(args.question)
    elif args.command == "evaluate":
        step5_evaluate()


if __name__ == "__main__":
    main()
