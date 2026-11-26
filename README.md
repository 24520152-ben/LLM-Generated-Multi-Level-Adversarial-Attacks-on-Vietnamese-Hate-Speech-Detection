# ViHSD Baselines and Adversarial Attacks

[Research proposal](project_proposal.md) — updated 26 November 2026.
Four clean baselines have completed one run each (seed 42). The validation attack
pilot is implemented; no attack or annotation results are available yet.

## Baseline results

Values are percentages. Encoder checkpoints were selected by the highest validation
macro-F1 across 10 epochs; SVM has no epoch.

| Model | Selected epoch | Validation macro-F1 | Test macro-F1 | Test accuracy |
|---|---:|---:|---:|---:|
| Linear SVM | — | 62.59 | 61.83 | 86.72 |
| PhoBERT-base | 3 | 67.46 | 64.10 | 87.53 |
| ViSoBERT | 1 | 68.33 | 66.67 | 87.46 |
| XLM-R-base | 5 | 66.22 | 64.90 | 85.72 |

Metrics come from local `artifacts/baselines/<model>/metrics.csv` files.
Full validation/test results are recorded in
[the proposal](project_proposal.md#61-clean-baselines). Raw artifacts remain ignored
by Git. These are single-seed clean scores, without confidence intervals; they do
not measure adversarial robustness. ViSoBERT is a suggested starting target because it
has the highest validation macro-F1 in these runs.

## Run on Kaggle

Start a Kaggle session with GPU and Internet enabled. Accept the access conditions
for [ViHSD](https://huggingface.co/datasets/uitnlp/vihsd) and attach a Hugging Face
read token to the notebook as a Kaggle secret named `HF_TOKEN`.

Push the project source to GitHub, then open the notebook for your baseline
in VS Code.

- [Linear SVM](scripts/train_svm.ipynb)
- [PhoBERT](scripts/train_phobert.ipynb)
- [ViSoBERT](scripts/train_visobert.ipynb)
- [XLM-R](scripts/train_xlmr.ipynb)

Select the Kaggle kernel and click **Run All**. Each notebook clones
`GITHUB_REPO_URL` into `/kaggle/working/ie403` on the first run and reuses that
checkout for later runs. Use a fresh Kaggle session to clone a newer revision.

All notebooks use the same dependency setup. After cloning, `uv sync --locked
--python 3.12` creates `.venv` inside the checkout with the versions recorded
in `uv.lock`. Training runs in a fresh process through `uv run --no-sync`.
Its output streams into the notebook. The first sync downloads Python 3.12
and packages as needed.

The notebook forwards the Kaggle `HF_TOKEN` secret and environment variables
to the training process, which uses the GPU specified in `BASELINE_OPTIONS`.
The project environment includes PhoBERT word segmentation and ViSoBERT
SentencePiece tokenizer dependencies.

Transformer notebooks run for 10 epochs and select the checkpoint with the
highest validation macro-F1. Set `"epochs": 1` in the first cell for a test run.

To train two models concurrently on two T4 GPUs, open two notebooks with
separate Python kernels on the same Kaggle server. Set `"device": "cuda:0"`
in one notebook and `"device": "cuda:1"` in the other. Each training run
prints its selected device and keeps its own checkpoint and CSV output.

## Run outputs

Each model saves two items directly under `artifacts/baselines/<model>/`.
Rerunning training replaces that model's checkpoint and metrics at the same paths.

- `checkpoint/` contains the model and the files needed by `load_classifier()`.
- `metrics.csv` has four columns (`split`, `epoch`, `macro_f1`, `accuracy`). A
  Transformer run writes one validation row per epoch and one test row for the
  selected epoch. SVM writes one row per split with an empty `epoch` column.

Baseline defaults are in `src/vihsd_attack/classifiers/config.py`. Run-specific overrides
are in the first cell of each notebook.

To regenerate per-example predictions, load `checkpoint/` with
`load_classifier()` and call `predict()` on texts from the original dataset
snapshot and split.

## Attack pilot on Kaggle

Attack settings are in `src/vihsd_attack/attacks/config.py`; shared labels and
artifact location remain in `src/vihsd_attack/config.py`.

Open the matching pilot notebook with a **T4 x2** kernel:

| Classifier | Notebook | Output directory |
|---|---|---|
| SVM | [pilot_svm.ipynb](scripts/pilot_svm.ipynb) | `artifacts/attacks/svm/` |
| PhoBERT | [pilot_phobert.ipynb](scripts/pilot_phobert.ipynb) | `artifacts/attacks/phobert/` |
| ViSoBERT | [pilot_visobert.ipynb](scripts/pilot_visobert.ipynb) | `artifacts/attacks/visobert/` |
| XLM-R | [pilot_xlmr.ipynb](scripts/pilot_xlmr.ipynb) | `artifacts/attacks/xlmr/` |

Each notebook points to its checkpoint under
`/kaggle/input/datasets/hophamsailam/vihsd-baselines/baselines/`.
Attach that dataset to the Kaggle notebook. The checkpoint directory must include
the saved preprocessing settings (or `model.joblib` and `baseline.json` for SVM).
The notebook clones and runs `uv sync --locked --python 3.12` just like training.
It reuses an existing checkout; use a fresh session or update that checkout first.

Qwen3-8B runs in **8-bit on cuda:0**, with FP16 computation and thinking disabled.
The encoder classifier uses **cuda:1**; SVM uses **CPU**. This is inference;
neither model is trained again. Run one pilot at a time in a shared GPU session.
Each notebook samples **5 correctly classified validation examples per label**
(15 per classifier), shared across its modes and beam widths. Eligibility is
classifier-specific, so the sampled source IDs can differ between classifiers.
This small pilot checks output quality and runtime and compares configurations
preliminarily; expand validation before making final configuration choices.

The default grid uses beam widths **1, 3, 5**, **Q=1–100 in steps of 1**, and
**G=5–100 in steps of 5** (`query_budgets` and `generation_budgets`). Each
example/mode/beam combination runs **once up to Q=100 and G=100**; its history
provides all 2,000 Q/G result rows without repeating generation or scoring.
Proposal order, per-call seeds, and prompts do not depend on lower checkpoints.
Eligibility scoring is a separate
preparation step, outside the attack query budget. A batch of five candidates
costs five queries; rejected/duplicate candidates cost zero.

The maximum of `generation_budgets` caps requested generation slots, shared
across all parents and, in MIX, all three generation modes. Each call consumes
its requested number of candidates before filtering, even for duplicates,
rejected candidates, missing entries or malformed responses. With five slots
per call, G=100 permits at most **20 LLM calls per search**. Intermediate G values
must be multiples of `candidates_per_call` so that their prefixes keep identical
generation requests. Only the largest G may end with a partial call, requesting
its remaining slots. Extra outputs are recorded as `extra` and
are not scored; this cap limits requested slots rather than guaranteeing how
many strings the LLM actually returns. Search ends when either budget is used
up or a round produces no new scored candidate. The generation cap is a pilot
cost-control choice, not a value established by prior papers. Rejected-output
audit rows describe the full trajectory. The console summarizes only the largest
Q/G pair for each mode/beam; all budget pairs are available in the result CSV.

MIX expands parents using all three modes and rotates their starting order across
examples and rounds. Edits accumulate along each branch. Automatic screening uses:

- **C:** NFC-normalized grapheme Levenshtein distance, at most two edits per step.
- **W:** at most `max_word_spans=2` diff regions, each bounded by
  `max_span_tokens=4` on both the parent and candidate sides. A unit is a Unicode
  letter/number run; it is not a segmented Vietnamese word. Insertions/deletions
  are allowed so that phrase expansion/contraction is not automatically rejected.
  A short whole-comment replacement can pass if it fits these bounds.
- **S:** changed lexical content only. There is no minimum number of diff regions
  and no automatic confirmation of paraphrasing or syntactic restructuring.

W/S reject changes involving only separators, punctuation or letter case. A small
character distance does not exclude a lexical replacement; the mode screens can
overlap. The span length four is an implementation default for validation pilot,
not a published Vietnamese threshold. W still depends on heuristic diff alignment
and can accept typos or unsuitable rewrites; S deliberately defers linguistic-level
judgment to annotation. Review both accepted and rejected candidates before freezing
these limits. `status=mode` means an automatic edit/form limit failed; `scored`
means a classifier evaluation, not a human-confirmed mode label.

There is no automatic semantic validator yet. Annotators must confirm the requested
edit level, meaning, target, offensiveness, and label preservation against the source
and, when checking a step, its parent. Searches continue past the first prediction
flip so that a rejected adversarial example does not hide later valid ones.

The shared generation prompt uses `VIHSD_LABEL_RUBRIC` in
[`constants.py`](src/vihsd_attack/constants.py), following the original
[ViHSD annotation guideline, Table 1](https://arxiv.org/abs/2103.11528):

| Label | Dataset-specific meaning |
|---|---|
| `CLEAN` (0) | No harassment or abusive content. |
| `OFFENSIVE` (1) | Offensive content or profanity without attacking a specific target. |
| `HATE` (2) | Abusive content targeting an individual or a group, including implicit or figurative attacks without explicit profanity. |

`HATE` is not restricted to protected groups. The prompt interprets slang and
obfuscated wording in context, without inventing targets or missing context.
Generation must preserve the supplied source label and meaning; the rubric does
not authorize relabeling or rewriting the source to fit a label. It guides
generation but does not certify candidate validity.

The four notebooks save to `artifacts/attacks/<model>/` via `output_dir`, so their
results do not overwrite one another. A direct `run_pilot()` call without an
`output_dir` still defaults to `artifacts/attacks/`. Rerunning a model's pilot
replaces its two CSV files at the same paths:

- **`candidates.csv`** has 9 columns: `id`, `mode`, `beam_width`, `query`,
  `parent_query`, `step_mode`, `candidate`, `prediction`, `status`.
  It keeps proposals and rejection reasons for reviewing the pilot filters, and
  all scored candidates for reconstructing ASR at each budget. `parent_query`
  references an earlier scored candidate in the same example/mode/beam search;
  zero means the original comment. `step_mode` and these links reconstruct the
  edit sequence without repeating parent text or storing a separate path.
  Empty query numbers mean the candidate was not scored. For an unparseable LLM
  response, the raw text appears once in `candidate`, with `status=invalid_output`.
- **`results.csv`** has 11 columns: `id`, `mode`, `beam_width`, `original`, `label`,
  `query_budget`, `generation_budget`, `queries_used`, `auto_success`, `llm_calls`,
  `seconds`. One row represents an example/mode/beam/Q/G, including failed searches.
  Original text and gold labels are stored here; match by `id` to inspect a
  candidate against its source. Predictions remain in the candidate file rather
  than being copied into results. `llm_calls` describes the joint Q/G prefix,
  including failed generation calls before that prefix ends. `seconds` describes
  the entire shared search and is filled only on its largest Q/G row; other
  budget pairs leave it empty. `queries_used` below
  the budget means the search stopped before reaching that query budget, either
  because no new candidates were scored or the generation-slot cap was reached.
  To reconstruct a pair's scored candidate pool, match `id`, `mode`, `beam_width`
  and select `status=scored` with `query <= queries_used`, rather than just Q.
  Thus the candidate CSV stores one trajectory without adding a G column.

Margins are used in memory to rank the beam and can be recomputed from the saved
classifier checkpoint. Candidate rows do not store margins or LLM call numbers.

`auto_success` is a classifier prediction flip, pending human validation. Review
all scored flips when checking whether a search found any valid adversarial example.

The final adversarial test set will retain **every distinct, validated successful
variant** within the test configurations and budgets fixed on validation. If one
source has seven valid successful variants, all seven are retained. Margin ranks
beam expansions; it does not select a single variant for the released dataset.
Deduplicate by source ID and NFC-normalized variant text, preserving provenance
and condition-specific validation from every contributing attack. ASR remains
source-based within each attack condition. For full-test before/after metrics,
each source contributes total weight one across its variants, with the original
retained when no valid successful variant exists. These are construction and
evaluation rules; annotation and final dataset export are not implemented yet.

The notebook is ready for a Kaggle run; full Qwen3-8B 8-bit inference has not yet
been verified on T4. These CSVs, once generated, are validation pilot outputs and
need human review before they can support valid attack-success claims.
Keep the notebook options and source revision with the experiment when reporting
results. Choose the shared configuration on validation before attacking test.

The query-budget landmarks 10/50/100 and width 5 follow
[TREPAT, EMNLP 2025, Sections 3.4–4](https://aclanthology.org/2025.emnlp-main.1405.pdf).
The denser Q/G checkpoints, widths 1/3, five candidates per call, and the per-step
edit/span limits are pilot choices.
Sampling follows [Qwen3's non-thinking guidance](https://huggingface.co/Qwen/Qwen3-8B).

Local CPU-only search checks live in the ignored `tests/` directory. If available, run:

```bash
PYTHONPATH=src uv run --no-sync python -m unittest discover -s tests -v
```
