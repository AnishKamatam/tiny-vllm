import torch
import torch.nn as nn

class LMHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    @property
    def weights(self) -> torch.Tensor:
        return self.lm_head.weight