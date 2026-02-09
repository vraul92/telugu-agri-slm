"""
Telugu Agricultural SLM - Model Architecture
Compact Transformer (~100-150M parameters)
"""
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE)"""
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000):
        super().__init__()
        
        # Calculate theta values
        theta = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        
        # Create position * theta matrix
        positions = torch.arange(max_seq_len)
        angles = torch.outer(positions, theta)
        
        # Cache sin and cos values
        self.register_buffer("cos_cached", angles.cos())
        self.register_buffer("sin_cached", angles.sin())
    
    def forward(self, x, seq_len: int):
        # x shape: (batch, heads, seq_len, head_dim)
        cos = self.cos_cached[:seq_len, :].to(x.device)
        sin = self.sin_cached[:seq_len, :].to(x.device)
        return self.apply_rotary_emb(x, cos, sin)
    
    def apply_rotary_emb(self, x, cos, sin):
        # x: (batch, heads, seq_len, head_dim)
        # Handle head_dim that might be odd
        head_dim = x.shape[-1]
        
        # Split x into pairs (only use even pairs)
        x1 = x[..., 0::2]  # Even indices
        x2 = x[..., 1::2]  # Odd indices
        
        # Ensure cos/sin match the split size
        cos = cos[:, :x1.shape[-1]]
        sin = sin[:, :x1.shape[-1]]
        
        # Add dimensions for broadcasting
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim//2)
        sin = sin.unsqueeze(0).unsqueeze(0)
        
        # Apply rotation
        rotated = torch.stack([-x2, x1], dim=-1)
        if rotated.shape[-1] == 2:
            rotated = rotated.flatten(-2)
        
        # Handle size mismatch
        if x1.shape[-1] != cos.shape[-1]:
            # Trim or pad cos/sin to match
            min_dim = min(x1.shape[-1], cos.shape[-1])
            x1 = x1[..., :min_dim]
            x2 = x2[..., :min_dim]
            cos = cos[..., :min_dim]
            sin = sin[..., :min_dim]
        
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        
        # Interleave back
        y = torch.stack([y1, y2], dim=-1).flatten(-2)
        
        # If original had odd dimension, pad back
        if y.shape[-1] < head_dim:
            padding = torch.zeros_like(y[..., :1])
            y = torch.cat([y, padding], dim=-1)
        elif y.shape[-1] > head_dim:
            y = y[..., :head_dim]
        
        return y


class MultiHeadAttention(nn.Module):
    """Multi-Head Self Attention with RoPE"""
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ):
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape to (batch, heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        q = self.rotary_emb(q, seq_len)
        k = self.rotary_emb(k, seq_len)
        
        # Scaled dot-product attention
        # Don't pass mask if causal (PyTorch handles it internally)
        if is_causal and mask is not None:
            mask = None
        
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            is_causal=is_causal if mask is None else False,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        output = self.o_proj(attn_output)
        
        return output


class FeedForward(nn.Module):
    """Gated Feed-Forward Network (SwiGLU)"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # SwiGLU: swish(x @ W1) * (x @ W3) @ W2
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    """Single Transformer Block"""
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        
        self.attn_norm = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout, max_seq_len)
        
        self.ff_norm = RMSNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        # Pre-norm architecture
        h = x + self.attn(self.attn_norm(x), mask=mask)
        out = h + self.ff(self.ff_norm(h))
        return out


class TeluguAgriSLM(nn.Module):
    """
    Telugu Agricultural Small Language Model
    ~110M parameters with default config
    """
    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        tie_weights: bool = True,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Token embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, max_seq_len)
            for _ in range(n_layers)
        ])
        
        # Final norm and output
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie input/output embeddings (reduces params, improves performance)
        if tie_weights:
            self.lm_head.weight = self.token_emb.weight
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Count parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model initialized with {n_params/1e6:.2f}M parameters")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        batch_size, seq_len = input_ids.shape
        
        # Embed tokens
        x = self.token_emb(input_ids)
        x = self.dropout(x)
        
        # Create causal mask if not provided
        if attention_mask is None:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device),
                diagonal=1
            ).bool()
        else:
            causal_mask = attention_mask
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, mask=causal_mask)
        
        # Final norm and projection
        x = self.norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1)
            )
        
        return {
            'loss': loss,
            'logits': logits,
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ):
        """Greedy or sampling generation"""
        self.eval()
        
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        for _ in range(max_new_tokens):
            # Forward pass
            with torch.no_grad():
                outputs = self.forward(input_ids)
                logits = outputs['logits'][:, -1, :]  # Last token logits
            
            # Apply temperature
            logits = logits / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token_id in set(input_ids[i].tolist()):
                        logits[i, token_id] /= repetition_penalty
            
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative prob > top_p
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                for i in range(batch_size):
                    indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                    logits[i, indices_to_remove] = -float('Inf')
            
            # Sample or take argmax
            if temperature > 0:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Append to input_ids
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Check for EOS
            if eos_token_id is not None:
                if (next_token == eos_token_id).all():
                    break
        
        return input_ids
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(vocab_size: int = 32000, **kwargs) -> TeluguAgriSLM:
    """Factory function to create model with default config"""
    return TeluguAgriSLM(vocab_size=vocab_size, **kwargs)


if __name__ == "__main__":
    # Test the model
    model = TeluguAgriSLM(
        vocab_size=32000,
        d_model=512,
        n_layers=8,
        n_heads=8,
        d_ff=2048,
        max_seq_len=512,
    )
    
    # Test forward pass
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, 32000, (batch_size, seq_len))
    labels = torch.randint(0, 32000, (batch_size, seq_len))
    
    outputs = model(input_ids, labels=labels)
    print(f"Loss: {outputs['loss'].item():.4f}")
    print(f"Logits shape: {outputs['logits'].shape}")
    
    # Test generation
    input_ids = torch.randint(0, 32000, (1, 10))
    generated = model.generate(
        input_ids,
        max_new_tokens=20,
        temperature=0.7,
        top_k=50,
    )
    print(f"Generated shape: {generated.shape}")
