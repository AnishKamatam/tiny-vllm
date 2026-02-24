import torch
import torch.nn.functional as F
from typing import Optional, Union

from .sampling_params import SamplingParams


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0.0:
        return logits
    return logits / temperature


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0:
        return logits

    top_k = min(top_k, logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, float("-inf"))
    return logits


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above threshold (keep at least one)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    # Scatter back to original indices
    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
    indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(indices_to_remove, float("-inf"))
    return logits


def sample(logits: torch.Tensor, params: SamplingParams) -> Union[int, torch.Tensor]:
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    # Greedy decoding - skip all the sampling overhead
    if params.temperature == 0.0:
        sampled = torch.argmax(logits, dim=-1)
        if squeeze_output:
            sampled = sampled.item()
        return sampled

    logits = apply_temperature(logits, params.temperature)

    if params.top_k > 0:
        logits = apply_top_k(logits, params.top_k)

    if params.top_p < 1.0:
        logits = apply_top_p(logits, params.top_p)

    probs = F.softmax(logits, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)

    if squeeze_output:
        sampled = sampled.item()

    return sampled


def sample_batch(logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
    return sample(logits, params)