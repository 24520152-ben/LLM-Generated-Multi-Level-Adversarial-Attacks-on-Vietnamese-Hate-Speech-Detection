"""Qwen3 inference on one GPU with 8-bit weights and thinking disabled."""

import json
from dataclasses import dataclass

from vihsd_attack.attacks.config import AttackConfig
from vihsd_attack.config import LABELS
from vihsd_attack.constants import (
    ATTACK_SYSTEM_PROMPT,
    ATTACK_USER_PROMPT,
    MODE_GUIDANCE,
)


@dataclass
class Generation:
    """One LLM response, including malformed output for pilot inspection."""

    candidates: list
    raw: str = ""
    cache_hit: bool = False
    invalid_output: bool = False


class CandidateGenerator:
    def __init__(self, config: AttackConfig, token: str | None = None):
        """Load Qwen on the requested CUDA device using LLM.int8().

        Args:
            config: Model, device, and generation settings.
            token: Optional Hugging Face read token.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        device = torch.device(config.llm_device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("8-bit generation requires an available CUDA GPU")
        self.config = config
        self._cache: dict[tuple, str] = {}
        self._cache_original: str | None = None
        options = {"revision": config.llm_revision, "token": token}
        self.tokenizer = AutoTokenizer.from_pretrained(config.llm_checkpoint, **options)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.llm_checkpoint,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map={"": str(device)},
            dtype=torch.float16,
            attn_implementation="sdpa",
            **options,
        ).eval()
        print(f"Generator: {config.llm_checkpoint}, 8-bit on {device}", flush=True)

    def generate(
        self,
        original: str,
        current: str,
        label: int,
        mode: str,
        seed: int,
        count: int,
    ) -> Generation:
        """Reuse identical seeded requests within the current source comment.

        Args:
            original: Unmodified source comment.
            current: Parent comment being expanded.
            label: Source ViHSD label ID.
            mode: C, W, or S for this step.
            seed: Reproducible seed for this individual LLM call.
            count: Requested candidates, bounded by the remaining generation slots.

        Returns:
            Parsed candidates, raw output, cache status, and top-level format validity.
        """
        import torch
        from transformers import set_seed

        config = self.config
        guidance = MODE_GUIDANCE[mode].format(
            max_char_edits=config.max_char_edits,
            max_word_spans=config.max_word_spans,
            max_span_tokens=config.max_span_tokens,
        )
        prompt = ATTACK_USER_PROMPT.format(
            count=count,
            mode=mode,
            guidance=guidance,
            data=json.dumps(
                {"original": original, "current": current, "label": LABELS[label]},
                ensure_ascii=False,
            ),
        )
        if original != self._cache_original:
            self._cache.clear()
            self._cache_original = original
        # The model/tokenizer are fixed for this instance; budgets and beam are not inputs.
        key = (
            ATTACK_SYSTEM_PROMPT,
            prompt,
            seed,
            config.temperature,
            config.top_p,
            config.top_k,
            config.max_new_tokens,
        )
        raw = self._cache.get(key)
        cache_hit = raw is not None
        if not cache_hit:
            inputs = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": ATTACK_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
                enable_thinking=False,
            ).to(self.model.device)
            set_seed(seed)
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    max_new_tokens=config.max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            new_tokens = output[0, inputs.input_ids.shape[1] :]
            raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            self._cache[key] = raw
        # Accept a JSON code fence, but never reinterpret prose as candidates.
        payload = raw
        if payload.startswith("```") and payload.endswith("```"):
            payload = payload.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            candidates = json.loads(payload)
        except json.JSONDecodeError:
            candidates = None
        invalid_output = not isinstance(candidates, list)
        return Generation(
            candidates=[] if invalid_output else candidates,
            raw=raw,
            cache_hit=cache_hit,
            invalid_output=invalid_output,
        )
