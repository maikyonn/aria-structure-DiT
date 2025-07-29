import math
from typing import Tuple
import torch
import torch.nn as nn
from lightning_utilities.core.imports import RequirementCache
from xformers.ops import SwiGLU
from src.fused_rotary_embedding import apply_rotary_emb_func

RoPECache = Tuple[torch.Tensor, torch.Tensor]
FlashAttention2Available = RequirementCache("flash-attn>=2.0.0.post1")

class TransEncoder(nn.Module):
    def __init__(self, vocab_size=32001, block_size=2048, n_layer=12, n_head=12, n_embd=768, 
                 n_query_groups=12, rotary_percentage=1.0, norm_eps=1e-5, bias=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.n_query_groups = n_query_groups
        self.rotary_percentage = rotary_percentage
        self.norm_eps = norm_eps
        self.bias = bias
        self.head_size = n_embd // n_head
        self.intermediate_size = 4 * n_embd  # For LLaMAMLP
        padded_vocab_size = vocab_size + (64 - vocab_size % 64) if vocab_size % 64 else vocab_size
        
        self.lm_head = nn.Linear(n_embd, padded_vocab_size, bias=False)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(padded_vocab_size, n_embd),
                h=nn.ModuleList(Block(self) for _ in range(n_layer)),
                ln_f=nn.LayerNorm(n_embd, eps=norm_eps),
            )
        )
        self.rope_cache = None
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / self.n_embd))
            elif isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / self.n_embd))
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            if isinstance(module, (LLaMAMLP, SelfAttention)) and hasattr(module, 'proj'):
                nn.init.normal_(module.proj.weight, mean=0.0, std=1 / math.sqrt(self.n_embd) / self.n_layer)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.size()
        assert T <= self.block_size, f"Cannot forward sequence of length {T}, block size is {self.block_size}"
        if self.rope_cache is None:
            self.rope_cache = self.build_rope_cache(idx)
        cos, sin = self.rope_cache
        cos, sin = cos[:T], sin[:T]
        x = self.transformer.wte(idx)
        for block in self.transformer.h:
            x = block(x, (cos, sin))
        x = self.transformer.ln_f(x)
        return self.lm_head(x)

    def build_rope_cache(self, idx: torch.Tensor) -> RoPECache:
        n_elem = int(self.rotary_percentage * self.head_size)
        theta = 1.0 / (10000 ** (torch.arange(0, n_elem, 2, device=idx.device) / n_elem))
        seq_idx = torch.arange(self.block_size, device=idx.device)
        idx_theta = torch.outer(seq_idx, theta)
        cos, sin = torch.cos(idx_theta), torch.sin(idx_theta)
        return cos.to(torch.bfloat16), sin.to(torch.bfloat16)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm_1 = nn.LayerNorm(config.n_embd, eps=config.norm_eps)
        self.attn = SelfAttention(config)
        self.norm_2 = nn.LayerNorm(config.n_embd, eps=config.norm_eps)
        self.mlp = LLaMAMLP(config)

    def forward(self, x: torch.Tensor, rope: RoPECache) -> torch.Tensor:
        x = x + self.attn(self.norm_1(x), rope)
        x = x + self.mlp(self.norm_2(x))
        return x

class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        self.attn = nn.Linear(config.n_embd, shape, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.config = config

    def forward(self, x: torch.Tensor, rope: RoPECache) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.attn(x)
        q_per_kv = self.config.n_head // self.config.n_query_groups
        total_qkv = q_per_kv + 2
        qkv = qkv.view(B, T, self.config.n_query_groups, total_qkv, self.config.head_size)
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=-2)
        q = q.reshape(B, T, -1, self.config.head_size)
        k = k.reshape(B, T, -1, self.config.head_size)
        v = v.reshape(B, T, -1, self.config.head_size)
        cos, sin = rope
        q = apply_rotary_emb_func(q, cos, sin, False, True)
        k = apply_rotary_emb_func(k, cos, sin, False, True)
        y = self.scaled_dot_product_attention(q, k, v)
        y = y.reshape(B, T, C)
        return self.proj(y)

    def scaled_dot_product_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        scale = 1.0 / math.sqrt(self.config.head_size)
        if FlashAttention2Available and q.device.type == "cuda" and q.dtype in (torch.float16, torch.bfloat16):
            return flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=scale, causal=False)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, scale=scale, is_causal=False
        )
        return y.transpose(1, 2)

class LLaMAMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.swiglu = SwiGLU(config.n_embd, config.intermediate_size, bias=False, _pack_weights=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.swiglu(x)