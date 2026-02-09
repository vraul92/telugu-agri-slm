# Telugu Agricultural Small Language Model (SLM)

A complete implementation of a Telugu Agricultural SLM from scratch, capable of answering farming-related queries in Telugu language.

## 🎯 Objective

Create a working Telugu Agricultural SLM (~100-150M parameters) that can:
- Answer agricultural questions in Telugu
- Provide farming advice and recommendations
- Understand Telugu agricultural terminology

## 📁 Project Structure

```
telugu-agri-slm/
├── data/                   # Data storage
│   ├── raw/               # Raw collected data
│   ├── processed/         # Processed train/val/test splits
│   └── cache/             # Cache files
├── models/                # Model storage
│   ├── tokenizer/         # Trained tokenizer
│   ├── checkpoints/       # Model checkpoints
│   └── logs/              # Training logs
├── src/                   # Source code
│   ├── config.py          # Configuration settings
│   ├── data_pipeline.py   # Data collection and processing
│   ├── tokenizer.py       # Telugu tokenizer training
│   ├── model.py           # SLM model architecture
│   ├── train.py           # Training pipeline
│   ├── inference.py       # Inference API
│   └── evaluate.py        # Evaluation module
├── pipeline.py            # Main pipeline orchestrator
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## 🏗️ Architecture

### Model Specifications
- **Architecture**: Transformer with RoPE (Rotary Position Embeddings)
- **Parameters**: ~110M (configurable)
- **Vocabulary**: 32,000 tokens
- **Context Length**: 512 tokens
- **Layers**: 8 transformer layers
- **Hidden Size**: 512
- **Attention Heads**: 8
- **FFN Dimension**: 2048
- **Normalization**: RMSNorm (Pre-norm)
- **Activation**: SwiGLU

### Key Components

1. **Custom Telugu Tokenizer**: SentencePiece BPE optimized for Telugu agricultural text
2. **Compact Transformer**: Efficient architecture suitable for edge deployment
3. **Synthetic Dataset**: Generated Telugu agricultural Q&A, documents, and conversations
4. **Training Pipeline**: Pre-training with causal language modeling
5. **Inference API**: FastAPI-based REST API with CLI interface

## 🚀 Quick Start

### Installation

```bash
# Clone and navigate to project
cd telugu-agri-slm

# Install dependencies
pip install -r requirements.txt
```

### Running the Full Pipeline

```bash
# Run complete pipeline (data → tokenizer → train → eval)
python pipeline.py all

# Quick mode (smaller model, faster training)
python pipeline.py all --quick
```

### Individual Steps

```bash
# 1. Data collection
python pipeline.py data

# 2. Train tokenizer
python pipeline.py tokenizer

# 3. Train model
python pipeline.py train

# 4. Run inference
python pipeline.py inference --question "వరి పంట గురించి చెప్పండి"

# 5. Evaluate
python pipeline.py evaluate
```

## 💬 Usage Examples

### Interactive CLI
```bash
python src/inference.py --mode interactive

# Example conversation:
# 🌾 మీ ప్రశ్న: వరి పంటకు ఎంత నీరు అవసరం?
# 🤖 సమాధానం: వరి పంటకు 7-10 రోజులకు ఒకసారి నీరు పెట్టాలి...
```

### API Server
```bash
python src/inference.py --mode api --port 8000

# Query the API:
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "మొక్కజొన్న ఎప్పుడు వేయాలి?"}'
```

### Python API
```python
from src.inference import TeluguAgriInference

inference = TeluguAgriInference(
    model_path="models/checkpoints/final/pytorch_model.bin",
    tokenizer_path="models/tokenizer/telugu_agri_spm.model"
)

# Answer a question
answer = inference.answer_question("పత్తి పంటలో తెగులు నివారణ ఎలా?")
print(answer)

# Generate from prompt
text = inference.generate("వ్యవసాయంలో", max_new_tokens=100)
print(text)
```

## 📊 Evaluation

The model is evaluated on:

1. **Perplexity**: Language modeling quality
2. **Q&A Accuracy**: Answer correctness (keyword overlap)
3. **Agricultural Knowledge**: Domain-specific coverage
4. **Telugu Language**: Script preservation and coherence

### Running Evaluation
```bash
python pipeline.py evaluate
```

### Example Results
```
Perplexity: 45.23
Q&A Accuracy: 32.5%
Agri Knowledge: 48.2%
Telugu Language: 85.7%
```

## 🔧 Configuration

Edit `src/config.py` to customize:

```python
# Model architecture
model.d_model = 512        # Hidden size
model.n_layers = 8         # Number of layers
model.n_heads = 8          # Attention heads

# Training
training.batch_size = 8
training.learning_rate = 5e-4
training.num_epochs = 3

# Inference
inference.temperature = 0.7
inference.max_new_tokens = 256
```

## 🧪 Development

### Testing Model Components
```bash
# Test tokenizer
python src/tokenizer.py

# Test model
python src/model.py

# Test data pipeline
python src/data_pipeline.py
```

### Adding Custom Data

1. Add your Telugu agricultural text to `data/raw/`
2. Update `src/data_pipeline.py` with new data sources
3. Re-run: `python pipeline.py data`

## 📈 Training Progress

The training pipeline saves:
- Checkpoints every N steps
- Best model based on validation loss
- Training logs and metrics
- Final model at `models/checkpoints/final/`

## 🛠️ Technical Details

### Tokenizer
- **Type**: SentencePiece BPE
- **Vocab Size**: 32,000
- **Coverage**: 99.95% for Telugu + English
- **Special Tokens**: `<pad>`, `<s>`, `</s>`, `<unk>`, `<mask>`, domain-specific tags

### Training
- **Objective**: Causal Language Modeling
- **Optimizer**: AdamW
- **Scheduler**: Linear warmup + Cosine annealing
- **Precision**: Mixed precision (FP16) when available
- **Gradient Clipping**: 1.0

### Supported Devices
- CUDA (NVIDIA GPUs)
- MPS (Apple Silicon)
- CPU (fallback)

## 📝 Limitations

- Uses synthetic training data (real data collection recommended for production)
- Model size is small (110M params) - suitable for edge deployment
- Context length limited to 512 tokens
- Evaluation metrics are heuristic-based

## 🔮 Future Improvements

1. Collect real Telugu agricultural data
2. Fine-tune on specific Q&A pairs
3. Add support for longer contexts
4. Implement instruction tuning
5. Add retrieval augmentation (RAG)
6. Quantization for edge deployment
7. Mobile app integration

## 📄 License

MIT License - See LICENSE file

## 🙏 Acknowledgments

- Inspired by TinyLlama, Phi-2, and other compact LLMs
- Telugu language resources from various open datasets
- Agricultural knowledge from Indian farming practices

## 📧 Contact

For questions or contributions, please open an issue on the repository.

---

**Note**: This is a proof-of-concept implementation. For production use, additional data collection, training, and evaluation are recommended.
