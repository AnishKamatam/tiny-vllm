# AGENTS.md

## Cursor Cloud specific instructions

This is a from-scratch LLaMA inference engine in pure Python/PyTorch. There are no services, databases, Docker, or web servers — it is a single-process Python library.

### Dependencies

- Install with `pip install -r requirements.txt` from the repo root.
- Core deps: `torch`, `transformers`, `safetensors`, `pytest`.

### Running / Testing

- **Import check:** `python3 -c "from llm import LLM"` (verifies all modules load).
- **Tests:** `python3 -m pytest` — no tests exist in the repo yet (exit code 5 is expected).
- **Lint:** No linter is configured. `python3 -m py_compile <file>` can check syntax.
- **Full inference** requires HuggingFace model weights (safetensors format) and preferably a CUDA GPU. Use `device="cpu"` for unit-level testing without GPU.

### Known issues

- `utils/sampling.py` was missing `from typing import Union` — this has been fixed on the `cursor/development-environment-setup-aad2` branch.

### Architecture notes

- `layers/` — individual neural network layers (Attention with GQA, RoPE, SwiGLU MLP, RMSNorm, Embedding, LMHead, TransformerBlock).
- `models/` — `LlamaConfig` (loads from HuggingFace config) and `LlamaModel` (assembles layers, loads safetensors weights).
- `utils/` — `Tokenizer` (wraps HuggingFace `AutoTokenizer`), `SamplingParams`, `sample()` function (temperature, top-k, top-p).
- `llm.py` — top-level `LLM` class providing `generate()` for text generation with KV cache.

### Testing components on CPU without model weights

Instantiate `LlamaModel` with a small `LlamaConfig` and random weights to test forward passes, autoregressive generation with KV cache, and sampling — no downloaded model required.
