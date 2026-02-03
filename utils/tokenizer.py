from typing import List, Optional
from transformers import AutoTokenizer


class Tokenizer:
    def __init__(self, model_path: str, trust_remote_code: bool = False):
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    @property
    def vocab_size(self) -> int:
        return len(self._tokenizer)

    @property
    def eos_token_id(self) -> Optional[int]:
        return self._tokenizer.eos_token_id

    @property
    def bos_token_id(self) -> Optional[int]:
        return self._tokenizer.bos_token_id

    @property
    def pad_token_id(self) -> int:
        if self._tokenizer.pad_token_id is not None:
            return self._tokenizer.pad_token_id
        if self._tokenizer.eos_token_id is not None:
            return self._tokenizer.eos_token_id
        raise ValueError("No pad_token_id or eos_token_id found in tokenizer")
