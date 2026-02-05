import torch
import torch.nn as nn
from typing import Optional, Tuple


class RoPE(nn.Module):
    def __init__(
        self,
        head_dim: int,
        base: float = 10000.0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _get_cos_sin(self, positions: torch.Tensor, device: torch.device, dtype: torch.dtype):
        angles = positions.unsqueeze(-1) * self.inv_freq.to(device)
        angles = torch.cat([angles, angles], dim=-1)
        return torch.cos(angles).to(dtype), torch.sin(angles).to(dtype)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_heads, seq_len, head_dim = q.shape

        if positions is None:
            positions = torch.arange(seq_len, device=q.device, dtype=torch.float32)
        
        cos, sin = self._get_cos_sin(positions, q.device, q.dtype)
        
        if cos.ndim == 2:
            cos = cos.unsqueeze(0).unsqueeze(1)
            sin = sin.unsqueeze(0).unsqueeze(1)
        else:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        def rotate_half(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)

        q_rotated = (q * cos) + (rotate_half(q) * sin)
        k_rotated = (k * cos) + (rotate_half(k) * sin)

        return q_rotated, k_rotated

    def apply_kv_cache(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: Optional[torch.Tensor] = None,
        v_cache: Optional[torch.Tensor] = None,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_len = q.size(2)
        positions = torch.arange(offset, offset + seq_len, device=q.device, dtype=torch.float32)
        q_rot, k_rot = self.forward(q, k, positions)
        
        if k_cache is not None and v_cache is not None:
            k_rot = torch.cat([k_cache, k_rot], dim=2)
            v_out = torch.cat([v_cache, v], dim=2)
        else:
            v_out = v
            
        return q_rot, k_rot, v_out