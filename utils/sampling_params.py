from dataclasses import dataclass
from typing import Optional, List


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 256
    stop: Optional[List[str]] = None
    ignore_eos: bool = False

    def __post_init__(self):
        if self.stop is None:
            self.stop = []
        
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < -1:
            raise ValueError("top_k must be >= -1")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
