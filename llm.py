import torch
from typing import List, Optional, Union

from models.llama_config import LlamaConfig
from models.llama_model import LlamaModel
from utils.tokenizer import Tokenizer
from utils.sampling import sample
from utils.sampling_params import SamplingParams


class LLM:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        trust_remote_code: bool = False,
    ):
        self.model_path = model_path
        self.device = device
        self.trust_remote_code = trust_remote_code

        self.config = LlamaConfig.from_hf_config(model_path, trust_remote_code)
        self.tokenizer = Tokenizer(model_path, trust_remote_code)
        self.model = LlamaModel(self.config)
        
        self.model.load_weights(model_path, device=device)
        self.model.to(device)
        self.model.eval()

    def generate(
        self,
        prompts: Union[str, List[str]],
        sampling_params: Optional[SamplingParams] = None,
    ) -> Union[str, List[str]]:
        if sampling_params is None:
            sampling_params = SamplingParams()

        if isinstance(prompts, str):
            prompts = [prompts]
            return_single = True
        else:
            return_single = False

        results = []
        for prompt in prompts:
            result = self._generate_single(prompt, sampling_params)
            results.append(result)

        return results[0] if return_single else results

    def _generate_single(
        self,
        prompt: str,
        params: SamplingParams,
    ) -> str:
        input_ids = self.tokenizer.encode(prompt)
        input_ids_tensor = torch.tensor([input_ids], device=self.device)

        generated_ids = input_ids.copy()
        kv_cache = None
        position_offset = 0

        stop_token_ids = set()
        if not params.ignore_eos and self.tokenizer.eos_token_id is not None:
            stop_token_ids.add(self.tokenizer.eos_token_id)
        
        stop_sequences_ids = []
        if params.stop:
            for stop_seq in params.stop:
                stop_ids = self.tokenizer.encode(stop_seq, add_special_tokens=False)
                stop_sequences_ids.append(stop_ids)

        for step in range(params.max_tokens):
            with torch.no_grad():
                if step == 0:
                    logits, kv_cache = self.model(
                        input_ids_tensor,
                        position_offset=position_offset,
                        kv_cache=None,
                        use_cache=True,
                    )
                    position_offset += input_ids_tensor.size(1)
                else:
                    next_token_tensor = torch.tensor([[next_token]], device=self.device)
                    logits, kv_cache = self.model(
                        next_token_tensor,
                        position_offset=position_offset,
                        kv_cache=kv_cache,
                        use_cache=True,
                    )
                    position_offset += 1

                last_logits = logits[0, -1, :]
                next_token = sample(last_logits, params)

                if isinstance(next_token, torch.Tensor):
                    next_token = next_token.item()

            generated_ids.append(next_token)

            if next_token in stop_token_ids:
                break

            if stop_sequences_ids:
                if self._check_stop_sequences(generated_ids, stop_sequences_ids):
                    break

        generated_text = self.tokenizer.decode(generated_ids[len(input_ids):])
        return generated_text

    def _check_stop_sequences(
        self,
        generated_ids: List[int],
        stop_sequences_ids: List[List[int]],
    ) -> bool:
        for stop_seq_ids in stop_sequences_ids:
            if len(generated_ids) >= len(stop_seq_ids):
                if generated_ids[-len(stop_seq_ids):] == stop_seq_ids:
                    return True
        return False