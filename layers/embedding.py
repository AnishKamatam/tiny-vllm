import torch
import torch.nn as nn
from typing import Optional

class TokenEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(
            vocab_size, 
            hidden_size, 
            padding_idx=padding_idx
        )
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)
    
    @property
    def weights(self) -> torch.Tensor:
        return self.embedding.weight
