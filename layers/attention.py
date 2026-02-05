import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from layers.rope import RoPE


class Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        rope_base: float = 10000.0,
        rope_base: float = 10000.0,
        bias: bool = False,
    ):
        super().__init__()
        assert hidden_size == num_heads * head_dim

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
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

        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(3, dim=2)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        positions = torch.arange(
            position_offset, position_offset + T, device=q.device, dtype=torch.float32
        )
        q, k = self.rope(q, k, positions)

        # Handle KV cache (after RoPE, so cached keys are already rotated)
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        new_kv_cache = (k, v) if use_cache else None

        is_causal = q.size(2) == k.size(2)

        attn_output = F.scaled_dot_product_attention(
            q, k, v, is_causal=is_causal
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, T, self.hidden_size)
        attn_output = self.out_proj(attn_output)

        return attn_output, new_kv_cache