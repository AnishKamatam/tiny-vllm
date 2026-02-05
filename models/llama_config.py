from dataclasses import dataclass
from transformers import AutoConfig

@dataclass
class LlamaConfig:
    vocab_size: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    num_hidden_layers: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    max_position_embeddings: int = 4096

    @classmethod
    def from_hf_config(cls, model_path:str, trust_remote_code:bool = False):
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        return cls(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            num_attention_heads=hf_config.num_attention_heads,
            num_key_value_heads=getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads),
            intermediate_size=hf_config.intermediate_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            rms_norm_eps=getattr(hf_config, "rms_norm_eps", 1e-6),
            rope_theta=getattr(hf_config, "rope_theta", 10000.0),
            max_position_embeddings=hf_config.max_position_embeddings,
        )

    def __post_init__(self):
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads})"
            )
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads
    
    @property
    def num_kv_heads(self) -> int:
        return self.num_key_value_heads