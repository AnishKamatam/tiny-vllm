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
        assert activation in {"swiglu", "gelu", "relu"}
        self.activation = activation
        
        if activation == "swiglu":
            self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=bias)
        else:
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "swiglu":
            gate_up = self.gate_up_proj(x)
            gate, up = gate_up.chunk(2, dim=-1)
            return self.down_proj(F.silu(gate) * up)
        elif self.activation == "gelu":
            return self.down_proj(F.gelu(self.gate_proj(x)))
        else:  
            return self.down_proj(F.relu(self.gate_proj(x)))
