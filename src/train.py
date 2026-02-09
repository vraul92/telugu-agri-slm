"""
Training Pipeline for Telugu Agricultural SLM
"""
import os
import json
import time
import math
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from model import TeluguAgriSLM
from tokenizer import TeluguAgriTokenizer
from config import config


class TeluguAgriDataset(Dataset):
    """Dataset for Telugu agricultural text"""
    
    def __init__(
        self,
        data_file: str,
        tokenizer: TeluguAgriTokenizer,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        with open(data_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into chunks
        self.examples = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]
        print(f"Loaded {len(self.examples)} examples from {data_file}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        text = self.examples[idx]
        
        # Tokenize
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        
        # Truncate or pad
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            # Pad
            pad_id = self.tokenizer.sp_model.pad_id() if self.tokenizer.sp_model else 0
            tokens = tokens + [pad_id] * (self.max_length - len(tokens))
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1 if token != 0 else 0 for token in tokens]
        
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(tokens, dtype=torch.long),
        }


class Trainer:
    """Trainer for Telugu Agricultural SLM"""
    
    def __init__(
        self,
        model: TeluguAgriSLM,
        tokenizer: TeluguAgriTokenizer,
        train_dataset: TeluguAgriDataset,
        val_dataset: Optional[TeluguAgriDataset] = None,
        config: Any = config.training,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        
        # Setup device
        self.device = self._get_device()
        self.model.to(self.device)
        
        # Setup data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )
        
        # Setup optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay,
        )
        
        # Setup learning rate scheduler
        total_steps = min(config.max_steps, len(self.train_loader) * config.num_epochs)
        
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=config.warmup_steps,
        )
        
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - config.warmup_steps,
            eta_min=config.min_lr,
        )
        
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[config.warmup_steps],
        )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Create output directories
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logging_dir = Path(config.logging_dir)
        self.logging_dir.mkdir(parents=True, exist_ok=True)
        
        # Mixed precision
        self.use_amp = config.mixed_precision and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        print(f"Trainer initialized:")
        print(f"  Device: {self.device}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  Mixed precision: {self.use_amp}")
        print(f"  Total training steps: {total_steps}")
    
    def _get_device(self):
        """Get the best available device"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(self.config.device)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step"""
        self.model.train()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Forward pass with mixed precision
        if self.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs['loss'] / self.config.gradient_accumulation_steps
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs['loss'] / self.config.gradient_accumulation_steps
            loss.backward()
        
        return loss.item() * self.config.gradient_accumulation_steps
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}")
        
        for step, batch in enumerate(pbar):
            loss = self.train_step(batch)
            epoch_loss += loss
            num_batches += 1
            
            # Gradient accumulation
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Clip gradients
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clipping,
                )
                
                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    pbar.set_postfix({
                        'loss': f"{loss:.4f}",
                        'lr': f"{lr:.2e}",
                        'step': self.global_step,
                    })
                
                # Evaluation
                if self.global_step % self.config.eval_steps == 0 and self.val_dataset:
                    val_metrics = self.evaluate()
                    print(f"\nStep {self.global_step}: Val loss = {val_metrics['loss']:.4f}")
                
                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint(f"checkpoint-{self.global_step}")
                
                # Check max steps
                if self.global_step >= self.config.max_steps:
                    break
        
        return {
            'train_loss': epoch_loss / num_batches,
        }
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set"""
        if self.val_dataset is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            total_loss += outputs['loss'].item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        perplexity = math.exp(avg_loss)
        
        # Save best model
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_checkpoint("best_model")
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity,
        }
    
    def train(self):
        """Main training loop"""
        print(f"\nStarting training for {self.config.num_epochs} epochs...")
        print(f"Training samples: {len(self.train_dataset)}")
        if self.val_dataset:
            print(f"Validation samples: {len(self.val_dataset)}")
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            print(f"\nEpoch {epoch + 1} complete:")
            print(f"  Train loss: {train_metrics['train_loss']:.4f}")
            
            # Evaluate
            if self.val_dataset:
                val_metrics = self.evaluate()
                print(f"  Val loss: {val_metrics['loss']:.4f}")
                print(f"  Val perplexity: {val_metrics['perplexity']:.2f}")
            
            # Save epoch checkpoint
            self.save_checkpoint(f"epoch-{epoch + 1}")
            
            # Check max steps
            if self.global_step >= self.config.max_steps:
                print(f"Reached max steps ({self.config.max_steps}), stopping training.")
                break
        
        total_time = time.time() - start_time
        print(f"\nTraining complete! Total time: {total_time/3600:.2f} hours")
        
        # Save final model
        self.save_checkpoint("final")
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint"""
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = checkpoint_dir / "pytorch_model.bin"
        torch.save(self.model.state_dict(), model_path)
        
        # Save config
        config_path = checkpoint_dir / "config.json"
        model_config = {
            'vocab_size': self.model.vocab_size,
            'd_model': self.model.d_model,
            'n_layers': len(self.model.layers),
            'n_heads': self.model.layers[0].attn.n_heads,
            'd_ff': self.model.layers[0].ff.w1.out_features,
            'max_seq_len': self.model.max_seq_len,
        }
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        
        # Save training state
        state = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
        }
        state_path = checkpoint_dir / "training_state.json"
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"Checkpoint saved to {checkpoint_dir}")
    
    def load_checkpoint(self, name: str):
        """Load model checkpoint"""
        checkpoint_dir = self.output_dir / name
        
        # Load model
        model_path = checkpoint_dir / "pytorch_model.bin"
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        # Load training state
        state_path = checkpoint_dir / "training_state.json"
        if state_path.exists():
            with open(state_path, 'r') as f:
                state = json.load(f)
            self.global_step = state['global_step']
            self.epoch = state['epoch']
            self.best_val_loss = state['best_val_loss']
        
        print(f"Checkpoint loaded from {checkpoint_dir}")


def main():
    """Main training function"""
    print("=" * 60)
    print("Telugu Agricultural SLM - Training Pipeline")
    print("=" * 60)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer_path = "models/tokenizer/telugu_agri_spm.model"
    if not os.path.exists(tokenizer_path):
        print(f"Tokenizer not found at {tokenizer_path}")
        print("Please train tokenizer first using: python src/tokenizer.py")
        return
    
    tokenizer = TeluguAgriTokenizer()
    tokenizer.load(tokenizer_path)
    print(f"Tokenizer loaded: vocab_size={tokenizer.sp_model.get_piece_size()}")
    
    # Create model
    print("\nCreating model...")
    model = TeluguAgriSLM(
        vocab_size=tokenizer.sp_model.get_piece_size(),
        d_model=512,
        n_layers=8,
        n_heads=8,
        d_ff=2048,
        max_seq_len=512,
    )
    
    # Load datasets
    print("\nLoading datasets...")
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
    
    # Create trainer
    trainer = Trainer(model, tokenizer, train_dataset, val_dataset)
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
