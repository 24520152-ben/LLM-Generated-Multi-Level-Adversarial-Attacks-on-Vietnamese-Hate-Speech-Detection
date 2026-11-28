"""Quantized, text-only judging with strict JSON output parsing."""

import json

from vihsd_attack.config import LABELS
from vihsd_attack.judges.config import JUDGE_MODELS, JudgeConfig
from vihsd_attack.judges.constants import JUDGE_SYSTEM_PROMPT, JUDGE_USER_PROMPT
from vihsd_attack.judges.data import INPUT_FIELDS

CRITERIA = ("label", "valid")
VOTE_FIELDS = [*INPUT_FIELDS, "candidate_label", "valid"]


def parse_verdict(raw: str) -> dict:
    """Validate the two-field JSON verdict; failed responses remain unresolved.

    Args:
        raw: Decoded response, optionally wrapped in a JSON code fence.

    Returns:
        Candidate label and combined validity, both UNSURE for malformed output.
    """
    payload = raw.strip()
    if payload.startswith("```") and payload.endswith("```"):
        payload = payload.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        value = json.loads(payload)
        if not isinstance(value, dict) or set(value) != set(CRITERIA):
            raise ValueError("Wrong fields")
        for key in CRITERIA:
            choices = {*LABELS, "UNSURE"} if key == "label" else {"YES", "NO", "UNSURE"}
            if value[key] not in choices:
                raise ValueError("Invalid category")
        return value
    except (ValueError, TypeError):
        return dict.fromkeys(CRITERIA, "UNSURE")


class LLMJudge:
    def __init__(self, config: JudgeConfig, token: str | None = None):
        """Load one model on one T4 using FP16 compute and SDPA.

        Args:
            config: Judge, GPU and generation settings.
            token: Optional Hugging Face read token.
        """
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        if not torch.cuda.is_available():
            raise ValueError("Quantized judging requires a CUDA GPU")
        self.config = config
        checkpoint, bits = JUDGE_MODELS[config.judge]
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, token=token)
        quantization = BitsAndBytesConfig(
            load_in_8bit=bits == 8,
            load_in_4bit=bits == 4,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        # AutoModelForCausalLM includes text compatibility for Qwen3.5 and Gemma 4.
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            device_map={"": config.device},
            dtype=torch.float16,
            attn_implementation="sdpa",
            quantization_config=quantization,
            token=token,
        ).eval()
        print(
            f"Judge: {checkpoint}, {bits}-bit on {config.device}",
            flush=True,
        )

    def judge(self, row: dict) -> dict:
        """Judge one complete path without exposing gold labels or model predictions.

        Args:
            row: Input row containing the original comment and readable edit path.

        Returns:
            Parsed verdict or an explicit unresolved response; inputs are never cut.
        """
        import torch
        from transformers import GenerationConfig

        config = self.config
        prompt = JUDGE_USER_PROMPT.format(
            max_char_edits=config.max_char_edits,
            max_word_spans=config.max_word_spans,
            max_span_tokens=config.max_span_tokens,
            data=json.dumps(
                {"original": row["original"], "path": row["path"]}, ensure_ascii=False
            ),
        )
        thinking = (
            {"thinking_mode": "off"}
            if config.judge == "sealion"
            else {"enable_thinking": False}
        )
        inputs = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            **thinking,
        )
        if inputs.input_ids.shape[1] > config.max_input_tokens:
            print(
                f"Input too long for {row.get('source_id', 'judge input')}; keeping UNSURE.",
                flush=True,
            )
            return dict.fromkeys(CRITERIA, "UNSURE")
        inputs = inputs.to(self.model.device)
        generation = GenerationConfig(
            do_sample=False,
            max_new_tokens=config.max_new_tokens,
            eos_token_id=self.model.generation_config.eos_token_id,
            pad_token_id=(
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            ),
        )
        with torch.inference_mode():
            output = self.model.generate(**inputs, generation_config=generation)
        raw = self.tokenizer.decode(
            output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
        ).strip()
        return parse_verdict(raw)
