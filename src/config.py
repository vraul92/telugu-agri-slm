"""
Configuration for Telugu Agricultural SLM
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    vocab_size: int = 32000
    d_model: int = 512          # Embedding dimension
    n_layers: int = 8           # Number of transformer layers
    n_heads: int = 8            # Number of attention heads
    d_ff: int = 2048            # Feed-forward dimension
    max_seq_len: int = 512      # Maximum sequence length
    dropout: float = 0.1
    attention_dropout: float = 0.1
    
    # SLM target: ~100-150M parameters
    # With these settings, we get ~110M params


@dataclass
class TokenizerConfig:
    """Tokenizer configuration"""
    vocab_size: int = 32000
    model_type: str = "bpe"     # "bpe" or "unigram"
    character_coverage: float = 0.9995  # For Telugu + English
    num_threads: int = 4
    max_sentence_length: int = 4096


@dataclass
class DataConfig:
    """Data configuration"""
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    cache_dir: str = "data/cache"
    
    # Data sources for Telugu agriculture
    data_sources: List[str] = field(default_factory=lambda: [
        "synthetic_telugu_agri",  # Primary source
    ])
    
    min_text_length: int = 10
    max_text_length: int = 10000
    train_split: float = 0.9
    val_split: float = 0.05
    test_split: float = 0.05


@dataclass
class TrainingConfig:
    """Training configuration"""
    output_dir: str = "models/checkpoints"
    logging_dir: str = "models/logs"
    
    # Training hyperparameters
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-4
    min_lr: float = 1e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    
    # Training schedule
    num_epochs: int = 3
    warmup_steps: int = 1000
    max_steps: int = 10000
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    
    # Optimization
    gradient_clipping: float = 1.0
    mixed_precision: bool = True
    
    # Device
    device: str = "auto"  # auto, cpu, cuda, mps


@dataclass
class InferenceConfig:
    """Inference configuration"""
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    do_sample: bool = True


@dataclass
class ProjectConfig:
    """Main project configuration"""
    project_name: str = "telugu-agri-slm"
    seed: int = 42
    
    model: ModelConfig = field(default_factory=ModelConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


# Global config instance
config = ProjectConfig()
