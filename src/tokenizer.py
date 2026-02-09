"""
Telugu Agricultural Tokenizer
Trains a custom BPE tokenizer optimized for Telugu agricultural text
"""
import os
import json
from typing import List, Optional
from pathlib import Path
import sentencepiece as spm
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors


class TeluguAgriTokenizer:
    """
    Custom tokenizer for Telugu agricultural text.
    Uses SentencePiece BPE with Telugu-specific settings.
    """
    
    # Special tokens (handled by SentencePiece defaults)
    PAD_TOKEN = "<pad>"
    EOS_TOKEN = "</s>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<s>"
    MASK_TOKEN = "<mask>"
    
    # Agricultural domain special tokens (user-defined, not overlapping with SP defaults)
    SPECIAL_TOKENS = [
        "<mask>",
        "<question>", "<answer>",
        "<crop>", "<pest>", "<disease>", "<fertilizer>",
        "<soil>", "<weather>", "<irrigation>", "<season>",
    ]
    
    def __init__(self, vocab_size: int = 32000, model_dir: str = "models/tokenizer"):
        self.vocab_size = vocab_size
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.sp_model = None
        self.tokenizer = None
        
    def train_sentencepiece(self, input_file: str, model_prefix: Optional[str] = None):
        """
        Train SentencePiece BPE tokenizer.
        
        Args:
            input_file: Path to text file for training
            model_prefix: Prefix for model files
        """
        if model_prefix is None:
            model_prefix = str(self.model_dir / "telugu_agri_spm")
        
        # Create user-defined symbols string
        user_symbols = ",".join(self.SPECIAL_TOKENS)
        
        # Training command
        train_args = {
            'input': input_file,
            'model_prefix': model_prefix,
            'vocab_size': self.vocab_size,
            'model_type': 'bpe',
            'character_coverage': 0.9995,
            'num_threads': 4,
            'split_digits': True,
            'allow_whitespace_only_pieces': True,
            'byte_fallback': True,
            'pad_id': 0,
            'eos_id': 1,
            'unk_id': 2,
            'bos_id': 3,
            'control_symbols': user_symbols,
            'max_sentence_length': 4096,
        }
        
        # Build command string
        cmd = f"--input={train_args['input']} " \
              f"--model_prefix={train_args['model_prefix']} " \
              f"--vocab_size={train_args['vocab_size']} " \
              f"--model_type={train_args['model_type']} " \
              f"--character_coverage={train_args['character_coverage']} " \
              f"--num_threads={train_args['num_threads']} " \
              f"--split_digits={str(train_args['split_digits']).lower()} " \
              f"--allow_whitespace_only_pieces={str(train_args['allow_whitespace_only_pieces']).lower()} " \
              f"--byte_fallback={str(train_args['byte_fallback']).lower()} " \
              f"--pad_id={train_args['pad_id']} " \
              f"--eos_id={train_args['eos_id']} " \
              f"--unk_id={train_args['unk_id']} " \
              f"--bos_id={train_args['bos_id']} " \
              f"--max_sentence_length={train_args['max_sentence_length']} " \
              f"--user_defined_symbols={user_symbols}"
        
        print(f"Training SentencePiece model...")
        spm.SentencePieceTrainer.train(cmd)
        
        # Load the trained model
        self.load(model_prefix + ".model")
        
        print(f"Tokenizer saved to {model_prefix}.model")
        print(f"Vocabulary size: {self.vocab_size}")
        
    def train_huggingface(self, files: List[str], save_path: Optional[str] = None):
        """
        Train using HuggingFace tokenizers library.
        
        Args:
            files: List of training files
            save_path: Where to save the tokenizer
        """
        if save_path is None:
            save_path = str(self.model_dir / "tokenizer.json")
        
        # Initialize BPE tokenizer
        tokenizer = Tokenizer(models.BPE(unk_token=self.UNK_TOKEN))
        
        # Pre-tokenizer: splits on whitespace and punctuation
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Whitespace(),
            pre_tokenizers.Punctuation(),
            pre_tokenizers.Digits(individual_digits=True),
        ])
        
        # Trainer
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.SPECIAL_TOKENS,
            min_frequency=2,
            show_progress=True,
        )
        
        # Train
        print(f"Training BPE tokenizer on {len(files)} files...")
        tokenizer.train(files, trainer)
        
        # Post-processing
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{self.BOS_TOKEN} $A {self.EOS_TOKEN}",
            pair=f"{self.BOS_TOKEN} $A {self.EOS_TOKEN} $B:1 {self.EOS_TOKEN}:1",
            special_tokens=[
                (self.BOS_TOKEN, tokenizer.token_to_id(self.BOS_TOKEN)),
                (self.EOS_TOKEN, tokenizer.token_to_id(self.EOS_TOKEN)),
            ],
        )
        
        # Decoder
        tokenizer.decoder = decoders.BPE()
        
        # Save
        tokenizer.save(save_path)
        self.tokenizer = tokenizer
        
        print(f"Tokenizer saved to {save_path}")
        
    def load(self, model_path: str):
        """Load a trained SentencePiece model"""
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(model_path)
        self.vocab_size = self.sp_model.get_piece_size()
        return self
        
    def load_hf(self, tokenizer_path: str):
        """Load a HuggingFace tokenizer"""
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.vocab_size = self.tokenizer.get_vocab_size()
        return self
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs"""
        if self.sp_model:
            if add_special_tokens:
                return self.sp_model.encode(text, add_bos=True, add_eos=True)
            else:
                return self.sp_model.encode(text, add_bos=False, add_eos=False)
        elif self.tokenizer:
            encoding = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
            return encoding.ids
        else:
            raise RuntimeError("Tokenizer not loaded")
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text"""
        if self.sp_model:
            if skip_special_tokens:
                return self.sp_model.decode(ids)
            else:
                pieces = [self.sp_model.id_to_piece(id) for id in ids]
                return "".join(pieces).replace("▁", " ")
        elif self.tokenizer:
            return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
        else:
            raise RuntimeError("Tokenizer not loaded")
    
    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Encode a batch of texts"""
        return [self.encode(text) for text in texts]
    
    def decode_batch(self, id_batches: List[List[int]]) -> List[str]:
        """Decode a batch of token IDs"""
        return [self.decode(ids) for ids in id_batches]
    
    def vocab_size(self) -> int:
        """Get vocabulary size"""
        if self.sp_model:
            return self.sp_model.get_piece_size()
        elif self.tokenizer:
            return self.tokenizer.get_vocab_size()
        return 0
    
    def save_vocab(self, output_file: str):
        """Save vocabulary to file"""
        if self.sp_model:
            with open(output_file, 'w', encoding='utf-8') as f:
                for i in range(self.sp_model.get_piece_size()):
                    piece = self.sp_model.id_to_piece(i)
                    f.write(f"{piece}\n")
    
    def get_special_tokens(self) -> dict:
        """Get special token IDs"""
        if self.sp_model:
            return {
                'pad_token_id': self.sp_model.pad_id(),
                'eos_token_id': self.sp_model.eos_id(),
                'bos_token_id': self.sp_model.bos_id(),
                'unk_token_id': self.sp_model.unk_id(),
            }
        return {}


def create_sample_training_data(output_file: str, num_samples: int = 10000):
    """
    Create synthetic training data for tokenizer.
    Mix of Telugu agricultural text patterns.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Telugu agricultural templates
    templates = [
        "వరి పంటకు ఎంత నీరు అవసరం?",
        "మొక్కజొన్న ఎప్పుడు వేయాలి?",
        "పత్తి పంటలో తెగులు నివారణ చర్యలు",
        "ఎరువుల వాడకం మరియు సమయం",
        "నేల సారాన్ని ఎలా పెంచాలి?",
        "వర్షాధార వ్యవసాయంలో మెరుగైన పద్ధతులు",
        "సేంద్రీయ వ్యవసాయం ప్రయోజనాలు",
        "పచ్చదనం పంటల నిర్వహణ",
        "నీటిపారుదల వ్యవస్థల అభివృద్ధి",
        "పంటల మార్పిడి ప్రణాళిక",
    ]
    
    telugu_words = [
        "వరి", "మొక్కజొన్న", "పత్తి", "పంట", "వ్యవసాయం", "ఎరువు", "నీరు",
        "నేల", "విత్తనాలు", "రైతు", "పొలం", "కూరగాయలు", "పండ్లు", "పశువులు",
        "దుక్కి", "నాట్లు", "కోత", "ధాన్యం", "అభివృద్ధి", "ఆదాయం", "సాగు",
    ]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(num_samples):
            # Generate varied sentences
            if i % 3 == 0:
                # Question format
                line = f"{templates[i % len(templates)]}\n"
            elif i % 3 == 1:
                # Statement format
                words = " ".join(telugu_words[i % len(telugu_words):i % len(telugu_words) + 5])
                line = f"{words} గురించి తెలుసుకోండి.\n"
            else:
                # Mixed Telugu-English
                line = f"{templates[i % len(templates)]} This is important for farmers.\n"
            
            f.write(line)
    
    print(f"Created sample training data: {output_file}")
    return output_file


if __name__ == "__main__":
    # Example usage
    import sys
    
    # Create sample data
    data_file = "data/tokenizer_train.txt"
    create_sample_training_data(data_file, num_samples=5000)
    
    # Train tokenizer
    tokenizer = TeluguAgriTokenizer(vocab_size=32000)
    tokenizer.train_sentencepiece(data_file)
    
    # Test
    test_text = "వరి పంటకు ఎంత నీరు అవసరం?"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"Original: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
