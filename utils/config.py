from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    model_path: str  # Path to model weights directory
    device: str = "cuda"  # Device to run inference on
    tensor_parallel_size: int = 1  # Number of GPUs for tensor parallelism
    enforce_eager: bool = False  # Use eager mode instead of CUDA graphs

    def __post_init__(self):
        if self.tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be >= 1")
        if self.device not in ["cuda", "cpu"]:
            raise ValueError("device must be 'cuda' or 'cpu'")
