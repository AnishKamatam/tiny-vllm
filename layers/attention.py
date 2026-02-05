import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .rope import RoPE


class Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rope_base: float = 10000.0,
        bias: bool = False,
    ):
        super().__init__()
        assert hidden_size == num_heads * head_dim

        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
            )
        if num_heads < num_kv_heads:
            raise ValueError(
                f"num_heads ({num_heads}) must be >= num_kv_heads ({num_kv_heads})"
            )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.rope = RoPE(head_dim, base=rope_base)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_offset: int = 0,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        B, T, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        positions = torch.arange(
            position_offset, position_offset + T, device=q.device, dtype=torch.float32
        )
        q, k = self.rope(q, k, positions)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        new_kv_cache = (k, v) if use_cache else None

        # Expand KV heads for GQA: repeat each KV head for its group of Q heads
        if self.num_kv_groups > 1:
            k = k.unsqueeze(2).expand(B, self.num_kv_heads, self.num_kv_groups, -1, self.head_dim)
            k = k.reshape(B, self.num_heads, -1, self.head_dim)
            v = v.unsqueeze(2).expand(B, self.num_kv_heads, self.num_kv_groups, -1, self.head_dim)
            v = v.reshape(B, self.num_heads, -1, self.head_dim)

        is_causal = q.size(2) == k.size(2)

        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, T, self.hidden_size)
        attn_output = self.out_proj(attn_output)

        return attn_output, new_kv_cache