import torch
import torch.nn as nn
from typing import List, Optional, Tuple
import os

from layers.embedding import TokenEmbedding
from layers.transformer_block import TransformerBlock
from layers.norm import RMSNorm
from layers.lmhead import LMHead
from .llama_config import LlamaConfig


class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = TokenEmbedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
        )

        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                intermediate_size=config.intermediate_size,
                mlp_activation="swiglu",
                norm_eps=config.rms_norm_eps,
                rope_base=config.rope_theta,
                bias=False,
            )
            for _ in range(config.num_hidden_layers)
        ])

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = LMHead(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_offset: int = 0,
        kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        hidden_states = self.embed_tokens(input_ids)

        if kv_cache is None:
            kv_cache = [None] * len(self.layers)

        new_kv_cache = []
        for i, layer in enumerate(self.layers):
            hidden_states, layer_kv_cache = layer(
                hidden_states,
                position_offset=position_offset,
                kv_cache=kv_cache[i],
                use_cache=use_cache,
            )
            if use_cache:
                new_kv_cache.append(layer_kv_cache)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits, new_kv_cache if use_cache else None

    @torch.no_grad()
    def load_weights(self, model_path: str, device: str = "cpu"):
        from safetensors import safe_open
        import glob
        import json

        # Find safetensor files (handles both single and sharded)
        safetensor_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))

        if not safetensor_files:
            raise FileNotFoundError(f"No safetensors files found in {model_path}")

        # Build weight name -> file mapping from index if sharded
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                index = json.load(f)
            weight_map = {k: os.path.join(model_path, v) for k, v in index["weight_map"].items()}
        else:
            # Single file - all weights in one file
            weight_map = None
            single_file = safetensor_files[0]

        def get_tensor(name: str) -> torch.Tensor:
            if weight_map:
                file_path = weight_map[name]
            else:
                file_path = single_file
            with safe_open(file_path, framework="pt", device=device) as f:
                return f.get_tensor(name)

        # Load embedding
        self.embed_tokens.embedding.weight.data.copy_(
            get_tensor("model.embed_tokens.weight")
        )

        # Load transformer layers
        for i in range(self.config.num_hidden_layers):
            prefix = f"model.layers.{i}"

            self.layers[i].attn_norm.weight.data.copy_(
                get_tensor(f"{prefix}.input_layernorm.weight")
            )

            self.layers[i].attention.q_proj.weight.data.copy_(
                get_tensor(f"{prefix}.self_attn.q_proj.weight")
            )
            self.layers[i].attention.k_proj.weight.data.copy_(
                get_tensor(f"{prefix}.self_attn.k_proj.weight")
            )
            self.layers[i].attention.v_proj.weight.data.copy_(
                get_tensor(f"{prefix}.self_attn.v_proj.weight")
            )
            self.layers[i].attention.out_proj.weight.data.copy_(
                get_tensor(f"{prefix}.self_attn.o_proj.weight")
            )

            self.layers[i].mlp_norm.weight.data.copy_(
                get_tensor(f"{prefix}.post_attention_layernorm.weight")
            )

            gate_weight = get_tensor(f"{prefix}.mlp.gate_proj.weight")
            up_weight = get_tensor(f"{prefix}.mlp.up_proj.weight")
            self.layers[i].mlp.gate_up_proj.weight.data.copy_(
                torch.cat([gate_weight, up_weight], dim=0)
            )
            self.layers[i].mlp.down_proj.weight.data.copy_(
                get_tensor(f"{prefix}.mlp.down_proj.weight")
            )

        # Load final norm and lm_head
        self.norm.weight.data.copy_(get_tensor("model.norm.weight"))
        self.lm_head.lm_head.weight.data.copy_(get_tensor("lm_head.weight"))
