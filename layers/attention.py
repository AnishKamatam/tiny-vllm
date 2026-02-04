import torch
import torch.nn as nn
from typing import Optional, Tuple


class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, head_dim, bias=False):
        super().__init__()
        assert hidden_size == num_heads * head_dim

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = 1.0 / (head_dim ** 0.5)

        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        B, T, _ = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(3, dim=2)  

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Handle KV cache
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        new_kv_cache = (k, v) if use_cache else None

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale


        if kv_cache is not None:
            cache_len = kv_cache[0].size(2)
            total_len = cache_len + T
            mask = torch.tril(
                torch.ones(1, 1, T, total_len, device=q.device, dtype=torch.bool),
                diagonal=cache_len,
            )
        else:
            mask = torch.tril(torch.ones(1, 1, T, T, device=q.device, dtype=torch.bool))

        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, T, self.hidden_size)
        attn_output = self.out_proj(attn_output)

        return attn_output, new_kv_cache