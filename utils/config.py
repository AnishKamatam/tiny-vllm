from dataclasses import dataclass


@dataclass
class ModelConfig:
    model_path: str
    device: str = "cuda"
    tensor_parallel_size: int = 1
    enforce_eager: bool = False

    def __post_init__(self):
        if self.tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be >= 1")
        if self.device not in ["cuda", "cpu"]:
            raise ValueError("device must be 'cuda' or 'cpu'")
