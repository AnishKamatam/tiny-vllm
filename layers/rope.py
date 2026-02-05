import torch
import torch.nn as nn
from typing import Optional, Tuple


class RoPE(nn.Module):
    def __init__(
        self,
        head_dim: int,
        base: float = 10000.0,
        base: float = 10000.0,
    ):
        """
        Implements Rotary Positional Embeddings (RoPE).
        
        Args:
            head_dim: The dimension of each attention head.
            max_seq_len: Maximum sequence length for pre-computed frequencies.
            base: The base for the exponential frequency scale.
        """
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Pre-compute inverse frequencies: 1 / (base ^ (2i / d))
        # Shape: (head_dim // 2)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _get_cos_sin(self, positions: torch.Tensor, device: torch.device, dtype: torch.dtype):
        """
        Calculates cos and sin components for given positions.
        """
        # angles shape: (batch, seq_len, head_dim // 2) or (seq_len, head_dim // 2)
        angles = positions.unsqueeze(-1) * self.inv_freq.to(device)
        
        # We repeat frequencies because RoPE is applied to pairs of features (x1, x2)
        # angles shape: (..., head_dim)
        angles = torch.cat([angles, angles], dim=-1)
        
        return torch.cos(angles).to(dtype), torch.sin(angles).to(dtype)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies RoPE to query and key tensors.
        
        Args:
            q: Query tensor of shape (batch, num_heads, seq_len, head_dim)
            k: Key tensor of shape (batch, num_heads, seq_len, head_dim)
            positions: Optional position indices of shape (batch, seq_len) or (seq_len,)
        
        Returns:
            Tuple of (rotated_q, rotated_k) with same shapes as input
        """
        batch_size, num_heads, seq_len, head_dim = q.shape

        if positions is None:
            positions = torch.arange(seq_len, device=q.device, dtype=torch.float32)
        
        cos, sin = self._get_cos_sin(positions, q.device, q.dtype)
        
        # Reshape cos/sin for broadcasting with (B, H, S, D)
        # Resulting shape: (B, 1, S, D) or (1, 1, S, D)
        if cos.ndim == 2:
            cos = cos.unsqueeze(0).unsqueeze(1)
            sin = sin.unsqueeze(0).unsqueeze(1)
        else:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        def rotate_half(x):
            # Split the last dimension into two halves
            x1, x2 = x.chunk(2, dim=-1)
            # [-x2, x1] is the standard rotation implementation
            return torch.cat((-x2, x1), dim=-1)

        # Apply the rotation: x_rotated = x*cos + rotate_half(x)*sin
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
        """
        Utility for rotating new tokens and concatenating with existing cache.
        
        Args:
            q, k, v: (batch, num_heads, new_tokens, head_dim)
            k_cache, v_cache: (batch, num_heads, historical_len, head_dim)
            offset: The current position in the sequence for the new tokens.
        """
        seq_len = q.size(2)
        
        # Calculate positions starting from the offset
        positions = torch.arange(offset, offset + seq_len, device=q.device, dtype=torch.float32)
        
        # Rotate ONLY the new keys and queries
        q_rot, k_rot = self.forward(q, k, positions)
        
        # Concatenate with cache if it exists
        if k_cache is not None and v_cache is not None:
            k_rot = torch.cat([k_cache, k_rot], dim=2)
            v_out = torch.cat([v_cache, v], dim=2)
        else:
            v_out = v
            
        return q_rot, k_rot, v_out