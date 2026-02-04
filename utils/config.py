from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ModelConfig:
    model_path: str  # Path to model weights directory
    device: Literal["cuda", "cpu"] = "cuda"  # Device to run inference on
    tensor_parallel_size: int = 1  # Number of GPUs for tensor parallelism
    enforce_eager: bool = False  # Use eager mode instead of CUDA graphs

    def __post_init__(self):
        if self.tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be >= 1")
