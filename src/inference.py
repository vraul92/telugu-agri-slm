"""
Inference API for Telugu Agricultural SLM
Simple API for querying the model with Telugu text
"""
import os
import json
import re
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from model import TeluguAgriSLM
from tokenizer import TeluguAgriTokenizer


@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    do_sample: bool = True


class TeluguAgriInference:
    """Inference engine for Telugu Agricultural SLM"""
    
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        device: Optional[str] = None,
    ):
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path)
        
        # Setup device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load tokenizer
        self._load_tokenizer()
        
        # Load model
        self._load_model()
        
        # Generation config
        self.gen_config = GenerationConfig()
        
        print("Inference engine ready!")
    
    def _load_tokenizer(self):
        """Load the tokenizer"""
        print(f"Loading tokenizer from {self.tokenizer_path}...")
        self.tokenizer = TeluguAgriTokenizer()
        self.tokenizer.load(str(self.tokenizer_path))
        
        # Get special token IDs
        self.special_tokens = self.tokenizer.get_special_tokens()
        self.pad_token_id = self.special_tokens.get('pad_token_id', 0)
        self.eos_token_id = self.special_tokens.get('eos_token_id', 1)
        self.bos_token_id = self.special_tokens.get('bos_token_id', 3)
    
    def _load_model(self):
        """Load the model"""
        print(f"Loading model from {self.model_path}...")
        
        # Load config
        config_path = self.model_path.parent / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                model_config = json.load(f)
        else:
            # Default config
            model_config = {
                'vocab_size': self.tokenizer.sp_model.get_piece_size(),
                'd_model': 512,
                'n_layers': 8,
                'n_heads': 8,
                'd_ff': 2048,
                'max_seq_len': 512,
            }
        
        # Create model
        self.model = TeluguAgriSLM(**model_config)
        
        # Load weights
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded: {self.model.count_parameters()/1e6:.2f}M parameters")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
    ) -> str:
        """
        Generate response for a given prompt
        
        Args:
            prompt: Input text in Telugu
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            repetition_penalty: Repetition penalty
        
        Returns:
            Generated text
        """
        # Use defaults if not specified
        max_new_tokens = max_new_tokens or self.gen_config.max_new_tokens
        temperature = temperature or self.gen_config.temperature
        top_k = top_k or self.gen_config.top_k
        top_p = top_p or self.gen_config.top_p
        repetition_penalty = repetition_penalty or self.gen_config.repetition_penalty
        
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                input_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_id=self.eos_token_id,
                pad_token_id=self.pad_token_id,
            )
        
        # Decode output (remove input tokens)
        generated_ids = output_ids[0, len(input_ids):].tolist()
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return response.strip()
    
    def answer_question(self, question: str, context: Optional[str] = None) -> str:
        """
        Answer an agricultural question in Telugu
        
        Args:
            question: Question in Telugu
            context: Optional context
        
        Returns:
            Answer in Telugu
        """
        # Format prompt
        if context:
            prompt = f"సందర్భం: {context}\n\nప్రశ్న: {question}\n\nసమాధానం:"
        else:
            prompt = f"ప్రశ్న: {question}\n\nసమాధానం:"
        
        return self.generate(prompt)
    
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Have a conversation with the model
        
        Args:
            messages: List of messages [{'role': 'user'/'assistant', 'content': '...'}]
        
        Returns:
            Model's response
        """
        # Format conversation
        formatted = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'user':
                formatted.append(f"వినియోగదారుడు: {content}")
            else:
                formatted.append(f"సహాయకుడు: {content}")
        
        prompt = "\n".join(formatted) + "\nసహాయకుడు:"
        
        return self.generate(prompt)
    
    def batch_generate(
        self,
        prompts: List[str],
        **generation_kwargs,
    ) -> List[str]:
        """Generate responses for multiple prompts"""
        return [self.generate(p, **generation_kwargs) for p in prompts]


# FastAPI application
app = FastAPI(
    title="Telugu Agricultural SLM API",
    description="API for Telugu agricultural question answering",
    version="1.0.0",
)

# Global inference engine
inference_engine: Optional[TeluguAgriInference] = None


# Request/Response models
class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.7
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.9
    repetition_penalty: Optional[float] = 1.1


class QuestionRequest(BaseModel):
    question: str
    context: Optional[str] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]


class Response(BaseModel):
    text: str
    status: str = "success"


@app.on_event("startup")
async def startup_event():
    """Initialize the inference engine on startup"""
    global inference_engine
    
    # Default paths
    model_path = os.getenv("MODEL_PATH", "models/checkpoints/final/pytorch_model.bin")
    tokenizer_path = os.getenv("TOKENIZER_PATH", "models/tokenizer/telugu_agri_spm.model")
    
    if not os.path.exists(model_path):
        print(f"Warning: Model not found at {model_path}")
        print("API will not be functional until model is loaded.")
        return
    
    inference_engine = TeluguAgriInference(model_path, tokenizer_path)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": inference_engine is not None,
    }


@app.post("/generate", response_model=Response)
async def generate(request: GenerateRequest):
    """Generate text from a prompt"""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        text = inference_engine.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
        )
        return Response(text=text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=Response)
async def ask_question(request: QuestionRequest):
    """Answer an agricultural question"""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        text = inference_engine.answer_question(
            question=request.question,
            context=request.context,
        )
        return Response(text=text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=Response)
async def chat(request: ChatRequest):
    """Chat with the model"""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        text = inference_engine.chat(messages)
        return Response(text=text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# CLI interface
class InferenceCLI:
    """Command-line interface for inference"""
    
    def __init__(self, model_path: str, tokenizer_path: str):
        self.engine = TeluguAgriInference(model_path, tokenizer_path)
    
    def interactive(self):
        """Run interactive CLI"""
        print("\n" + "=" * 60)
        print("Telugu Agricultural SLM - Interactive Mode")
        print("=" * 60)
        print("Type your questions in Telugu or English")
        print("Type 'quit' or 'exit' to stop")
        print("=" * 60 + "\n")
        
        while True:
            try:
                # Get input
                user_input = input("\n🌾 మీ ప్రశ్న: ").strip()
                
                # Check for exit
                if user_input.lower() in ['quit', 'exit', 'బయటకు']:
                    print("\nధన్యవాదాలు! వీడ్కోలు.")
                    break
                
                if not user_input:
                    continue
                
                # Generate response
                print("\n💭 ఆలోచిస్తున్నాను...")
                response = self.engine.answer_question(user_input)
                
                # Display response
                print(f"\n🤖 సమాధానం: {response}")
                
            except KeyboardInterrupt:
                print("\n\nధన్యవాదాలు! వీడ్కోలు.")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def run_once(self, question: str):
        """Run a single query"""
        response = self.engine.answer_question(question)
        print(f"ప్రశ్న: {question}")
        print(f"సమాధానం: {response}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Telugu Agricultural SLM Inference")
    parser.add_argument("--model", default="models/checkpoints/final/pytorch_model.bin",
                       help="Path to model checkpoint")
    parser.add_argument("--tokenizer", default="models/tokenizer/telugu_agri_spm.model",
                       help="Path to tokenizer")
    parser.add_argument("--mode", choices=["api", "interactive", "once"], default="interactive",
                       help="Run mode")
    parser.add_argument("--port", type=int, default=8000,
                       help="API server port")
    parser.add_argument("--question", type=str, help="Question for 'once' mode")
    
    args = parser.parse_args()
    
    if args.mode == "api":
        # Run API server
        print(f"Starting API server on port {args.port}...")
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    
    elif args.mode == "interactive":
        # Run interactive CLI
        cli = InferenceCLI(args.model, args.tokenizer)
        cli.interactive()
    
    elif args.mode == "once":
        # Run single query
        if not args.question:
            print("Error: --question required for 'once' mode")
            return
        cli = InferenceCLI(args.model, args.tokenizer)
        cli.run_once(args.question)


if __name__ == "__main__":
    main()
