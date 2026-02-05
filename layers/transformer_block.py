import torch
import torch.nn as nn
from typing import Literal, Optional, Tuple

from .attention import Attention
from .mlp import MLP
from .norm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        mlp_activation: Literal["swiglu", "gelu", "relu"] = "swiglu",
        norm_eps: float = 1e-6,
        rope_base: float = 10000.0,
        bias: bool = False,
    ):
        super().__init__()

        self.attention = Attention(
            hidden_size, num_heads, num_kv_heads, head_dim, rope_base=rope_base, bias=bias
        )
        self.mlp = MLP(hidden_size, intermediate_size, activation=mlp_activation, bias=bias)
        self.attn_norm = RMSNorm(hidden_size, eps=norm_eps)
        self.mlp_norm = RMSNorm(hidden_size, eps=norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_offset: int = 0,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)

        attn_output, kv_cache = self.attention(
            hidden_states,
            position_offset=position_offset,
            kv_cache=kv_cache,
            use_cache=use_cache,
        )

        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output

        return hidden_states, kv_cache
