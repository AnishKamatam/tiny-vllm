import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "swiglu",
        bias: bool = False,
    ):
        super().__init__()
        
        if activation not in {"swiglu", "gelu", "relu"}:
            raise ValueError(
                f"Unsupported activation: {activation}. "
                "Must be one of 'swiglu', 'gelu', or 'relu'."
            )
        
        self.activation = activation
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        
        if activation == "swiglu":
            self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=bias)
            self.act_fn = F.silu
        else:
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
            self.act_fn = {"gelu": F.gelu, "relu": F.relu}[activation]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "swiglu":
            gate_up = self.gate_up_proj(x)
            gate, up = gate_up.chunk(2, dim=-1)
            activated_x = self.act_fn(gate) * up
        else:
            activated_x = self.act_fn(self.up_proj(x))
        return self.down_proj(activated_x)
