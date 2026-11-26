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
    ) -> Generation:
        """Generate candidates without using the query budget in the prompt.

        Args:
            original: Unmodified source comment.
            current: Parent comment being expanded.
            label: Source ViHSD label ID.
            mode: C, W, or S for this step.
            seed: Reproducible seed for this individual LLM call.

        Returns:
            Parsed JSON candidates and raw output for inspecting format failures.
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
            count=config.candidates_per_call,
            mode=mode,
            guidance=guidance,
            data=json.dumps(
                {"original": original, "current": current, "label": LABELS[label]},
                ensure_ascii=False,
            ),
        )
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
        # Accept a JSON code fence, but never reinterpret prose as candidates.
        payload = raw
        if payload.startswith("```") and payload.endswith("```"):
            payload = payload.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            candidates = json.loads(payload)
        except json.JSONDecodeError:
            candidates = []
        return Generation(candidates if isinstance(candidates, list) else [], raw)
