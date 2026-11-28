# ViHSD Baselines and Adversarial Attacks

[Research proposal](project_proposal.md) — updated 29 November 2026.
Four clean baselines have completed one run each (seed 42). Validation attack
pilot outputs and three-LLM judge results are available for all four classifiers.
The current version uses LLM-judged validity; human review and judge-model upgrades
are deferred to a later version.

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

## Rerun validation attacks on Kaggle

Attack settings are in `src/vihsd_attack/attacks/config.py`; shared labels and
artifact location remain in `src/vihsd_attack/config.py`.

Open the matching attack notebook with a **T4 x2** kernel:

| Classifier | Notebook | Output directory |
|---|---|---|
| SVM | [attack_validation_svm.ipynb](scripts/attack_validation_svm.ipynb) | `artifacts/attacks/svm/` |
| PhoBERT | [attack_validation_phobert.ipynb](scripts/attack_validation_phobert.ipynb) | `artifacts/attacks/phobert/` |
| ViSoBERT | [attack_validation_visobert.ipynb](scripts/attack_validation_visobert.ipynb) | `artifacts/attacks/visobert/` |
| XLM-R | [attack_validation_xlmr.ipynb](scripts/attack_validation_xlmr.ipynb) | `artifacts/attacks/xlmr/` |

Each notebook points to its checkpoint under
`/kaggle/input/datasets/hophamsailam/vihsd-baselines/baselines/`.
Attach that dataset to the Kaggle notebook. The checkpoint directory must include
the saved preprocessing settings (or `model.joblib` and `baseline.json` for SVM).
The notebook clones and runs `uv sync --locked --python 3.12` in `.venv`.
It reuses an existing checkout; use a fresh session or update that checkout first.

The rerun uses the official **Qwen/Qwen3-14B** checkpoint, quantized to **4-bit
NF4** with double quantization, FP16 compute, SDPA and thinking disabled.
The generator runs on **cuda:0** and the encoder classifier on **cuda:1**;
SVM uses **CPU**. Run one notebook at a time. Neither model is trained again.
The original weight files total approximately **27.51 GiB**; they are downloaded
before quantization. This is disk usage, not the quantized model's VRAM footprint.
Caches and the Python environment require additional disk space.

The notebooks set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before starting
the attack process to reduce CUDA allocator fragmentation during loading.
After a failed 30B attempt, stop the old process and remove its cached weights or
start a fresh Kaggle session before downloading the 14B checkpoint. Restarting
only the notebook kernel does not remove cached weight files.

Each notebook samples **10 correctly classified validation examples per label**
(30 per classifier) with seed 42, shared across its modes and beam widths. Eligibility is
classifier-specific, so the sampled source IDs can differ between classifiers.
This reruns the validation experiments with the new generator; it does not attack
test. Existing results below describe the earlier Qwen3-8B run, not this rerun.

The default grid uses beam widths **1, 3, 5**, **Q=1–25 in steps of 1**, and
**G=5–25 in steps of 5** (`query_budgets` and `generation_budgets`). Each
example/mode/beam combination runs **once up to Q=25 and G=25**; its history
provides all 125 Q/G snapshots without repeating generation or scoring.
There are **360 searches per classifier** (30 sources × 4 modes × 3 beam widths).
Proposal order, per-call seeds, and prompts do not depend on lower checkpoints.
Eligibility scoring is a separate
preparation step, outside the attack query budget. A batch of five candidates
costs five queries; rejected/duplicate candidates cost zero.

The maximum of `generation_budgets` caps requested generation slots, shared
across all parents and, in MIX, all three generation modes. Each call consumes
its requested number of candidates before filtering, even for duplicates,
rejected candidates, missing entries or malformed responses. With five slots
per call, G=25 permits at most **5 generation requests per search**, including
cache hits. Intermediate G values
must be multiples of `candidates_per_call` so that their prefixes keep identical
generation requests. Only the largest G may end with a partial call, requesting
its remaining slots. Extra outputs are recorded as `extra` and
are not scored; this cap limits requested slots rather than guaranteeing how
many strings the LLM actually returns. Search ends when either budget is used
up or a round produces no new scored candidate. The generation cap is a pilot
cost-control choice, not a value established by prior papers. Rejected-output
audit rows describe the full trajectory. The console summarizes only the largest
Q/G pair for each mode/beam; all budget pairs are available in the result CSV.

The generator caches raw responses **in RAM for the current source comment**.
An identical system/user prompt, seed and decoding settings reuse the response
across beam widths. The loaded LLM/tokenizer stay fixed for that cache instance.
Malformed or empty responses are cached too. Returned candidates are parsed anew
so callers cannot modify the cached value. Cache entries are cleared when moving
to another source and are not saved to disk or shared between notebook runs.
Hits consume the same G slots as misses and follow the same classifier Q rules;
seeds still use request order. Beam order remains 1/3/5: whichever width runs first
pays for shared requests. Warm-cache timings and actual LLM calls depend on run
order, so report the order/cache policy when comparing compute costs.

MIX expands parents using all three modes and rotates their starting order across
examples and rounds. Edits accumulate along each branch. Automatic screening uses:

- **C:** NFC-normalized grapheme Levenshtein distance, one or two edits per step.
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
judgment to the post-search LLM judges. Structural-screen and judge acceptance
rates are reported separately. `status=mode` means an automatic edit/form limit failed; `scored`
means a classifier evaluation, not a human-confirmed mode label.

There is no semantic validator inside the search. The three post-search LLM judges
assess edit level, meaning and label preservation against the source and the full
path. Human review is deferred in the current version. Searches continue past the first prediction
flip so that a rejected adversarial example does not hide later valid ones.

C guidance was revised after the uncached pilot mostly returned unchanged text.
The prompt now requires a nonidentity edit, explains whole-comment grapheme costs
(including two edits for swapping adjacent characters), and supplies two synthetic
Vietnamese demonstrations in the same JSON-array format as the requested output.
All ten example candidates pass the existing C screen even at a one-edit limit.
It permits local spelling/diacritic changes only when the intended reading is
preserved, and allows fewer candidates rather than padding with unchanged copies.
This adapts typo/few-shot guidance from
[PromptAttack, ICLR 2024, Sections 3.1/3.3](https://proceedings.iclr.cc/paper_files/paper/2024/file/0c72285e193ec90dca93258128698cfb-Paper-Conference.pdf)
and operations from [TextBugger, NDSS 2019, Section III-C](https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-5_Li_paper.pdf).
The examples/counting policy are project choices, not Vietnamese thresholds
validated by those papers. Filters and shared prompts are unchanged; W/S wording is preserved but now
uses multiline formatting;
both C and MIX need fresh model evaluation under the revised C prompt. Existing
pilot CSVs still describe the previous prompt. Syntax/example checks do not prove
that Qwen3 follows the new instructions or that its candidates preserve meaning.

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
`output_dir` still defaults to `artifacts/attacks/`. Rerunning a model's attack
replaces its two CSV files at the same paths:

- **`candidates.csv`** has 12 columns: `source_id`, `original`, `original_label`,
  `parent`, `candidate`, `predicted_label`, `mode`, `step_mode`, `beam_width`,
  `query`, `generation_used`, `status`. Each row is a proposal or an invalid/missing
  output position. Original text, gold label and parent text are included so a
  row can be reviewed without joining another file. `predicted_label` is the
  classifier's prediction for the variant. It and `query` are empty when the
  candidate was not scored. `query` numbers restart for each source/mode/beam
  search; `generation_used` counts cumulative requested slots at the generating
  call, including filtered, missing and cached positions. Candidates from the
  same call share that value. To recover a scored pool for Q/G, match source,
  mode and beam, then filter `status=scored`, `query <= Q` and
  `generation_used <= G`. Parent text links to the original or an earlier scored
  candidate in the same search, allowing the complete edit path to be rebuilt.
  Status values are `scored`, `format`, `mode`, `duplicate`, `missing`,
  `invalid_output`, `extra`, and `unscored`. A valid empty array produces
  `missing` positions; malformed/empty/non-array output produces one
  `invalid_output` row retaining the raw response, followed by `missing` rows.
  Invalid array entries are `format`. None of these rejected positions consumes
  a classifier query. Excess items are `extra` and add neither Q nor G;
  `unscored` means a usable candidate could not be scored within the remaining Q.
- **`results.csv`** has 10 columns: `mode`, `beam_width`, `query_budget`,
  `generation_budget`, `samples`, `flips`, `queries`, `llm_calls`, `cache_hits`,
  `seconds`. Each row aggregates the completed source searches for one
  mode/beam/Q/G configuration, including failures in `samples`. `flips` counts
  sources with at least one classifier prediction flip, pending validity review;
  it does not count variants. `queries`, `llm_calls` and `cache_hits` are totals
  across those sources. Calls are actual LLM inferences; hits are reused
  responses. `seconds` sums full-trajectory runtime and is filled only at the
  largest Q/G pair. Costs from different budget rows must not be added together:
  they describe overlapping prefixes of the same searches. The file is refreshed
  after each completed search, with `samples` showing its current denominator.
  The rerun grid has 1,500 aggregate rows per model. The earlier Q/G=100 grid
  produced 24,000 rows per model. All four existing model outputs were converted without
  rerunning inference; generation positions were recovered from five-slot call
  boundaries and checked against the original Q/G snapshots.

Margins are used in memory to rank the beam and can be recomputed from the saved
classifier checkpoint. Review all scored prediction flips when checking whether
a search found any valid adversarial example.

The final adversarial test set will retain **every distinct, validated successful
variant** within the test configurations and budgets fixed on validation. If one
source has seven valid successful variants, all seven are retained. Margin ranks
beam expansions; it does not select a single variant for the released dataset.
Deduplicate by source ID and NFC-normalized variant text, preserving provenance
and condition-specific validation from every contributing attack. ASR remains
source-based within each attack condition. For full-test before/after metrics,
each source contributes total weight one across its variants, with the original
retained when no valid successful variant exists. These are construction and
evaluation rules. LLM judging has run on the pilot; final dataset export is not
implemented yet, and human review is deferred.

Current pilot CSVs for all four classifiers include request-cache accounting.
Earlier uncached outputs used a different prompt and are not a controlled timing
comparison. Current results are reported as LLM-judged ASR, without human-validation claims.
Keep the notebook options and source revision with the experiment when reporting
results. Choose the shared configuration on validation before attacking test.

The query-budget landmarks 10/50/100 and width 5 follow
[TREPAT, EMNLP 2025, Sections 3.4–4](https://aclanthology.org/2025.emnlp-main.1405.pdf).
The denser Q/G checkpoints, widths 1/3, five candidates per call, and the per-step
edit/span limits are pilot choices.
Sampling follows [Qwen3-14B non-thinking guidance](https://huggingface.co/Qwen/Qwen3-14B).
The rerun's Q/G maximum of 25 and ten sources per label are project choices.

Local CPU-only search checks live in the ignored `tests/` directory. If available, run:

```bash
PYTHONPATH=src uv run --no-sync python -m unittest discover -s tests -v
```

## LLM judges on Kaggle

Only **prediction-flipping** candidates enter this stage. Prepare the portable
input file locally after downloading the four attack outputs:

```bash
PYTHONPATH=src uv run python -c 'from vihsd_attack.judges.data import prepare_inputs; prepare_inputs("artifacts/attacks")'
```

This writes `artifacts/judges/judge_inputs.csv`. The current pilot contains
**1,570 distinct source/variant pairs and 1,773 distinct edit paths**. Paths include
non-flipping ancestors as context; those ancestors are not separate annotation
samples. Paths shared across classifiers/beam widths are judged once. Attack
provenance remains in the candidate CSVs. Different paths to one variant remain separate
because mode validity depends on the complete chain, especially for MIX.
This export uses all saved pilot searches; lock the source runs/configurations
before using the same procedure for a final test dataset.

Upload **that one CSV** as a Kaggle Dataset, attach it, and open
[scripts/judge.ipynb](scripts/judge.ipynb) with **T4 x2** and Internet enabled.
The notebook finds the unique `judge_inputs.csv` under `/kaggle/input`, or accepts
an explicit `INPUT_CSV`. It clones the repository and runs `uv sync --locked`
using the same isolated Python 3.12 workflow as the other notebooks. No classifier
checkpoint is needed. A fresh session clones newer source; existing clones must
be updated before running newly committed code.

| Stage | Checkpoint | Quantization | Default device |
|---|---|---|---|
| Independent judge 1 | `Qwen/Qwen3.5-9B` | 8-bit | `cuda:0` |
| Independent judge 2 | `aisingapore/Llama-SEA-LION-v3.5-8B-R` | 8-bit | `cuda:1` |
| Disagreement/uncertainty judge | `google/gemma-4-12B-it` | 4-bit NF4 | `cuda:0` |

The first two judges run in separate processes. After both exit, Gemma judges
only disagreements, `UNSURE` answers or malformed/overlong first-stage inputs.
Gemma does not see the first-stage answers. All models use text-only inputs,
FP16 computation, SDPA and thinking disabled. Greedy decoding, 128 output tokens
and an 8,192 input-token guard are implementation defaults, not validated optimal
settings. Overlong inputs stay unresolved; comments and paths are never silently
truncated. All three quantized checkpoints completed the pilot on Kaggle T4 GPUs.

Settings are in `src/vihsd_attack/judges/config.py` and prompts in
`src/vihsd_attack/judges/constants.py`. Judges return the final label and one
combined assessment of meaning preservation and mode compliance of every step.
They see neither gold source labels, classifier predictions, attack model names
nor other judges' answers. Code then compares the judge's label with the source
label: final `valid=YES` requires meaning, mode and label preservation together.
A definite failure gives `NO`; uncertainty, malformed JSON or an overlong input
stays `UNSURE`. Benchmark support for Vietnamese does not establish judge accuracy;
human calibration and judge-model upgrades are deferred in this version.

Outputs stay flat under `artifacts/judges/`:

- `judge_inputs.csv` has five columns: `source_id`, `original`, `original_label`,
  `candidate`, `path`. It contains no generated ID or attack-provenance JSON.
  `path` is one readable line, for example
  `1. [W] intermediate comment -> 2. [S] final variant`.
  Each numbered step records the requested mode and resulting text. Literal
  newlines/backslashes inside comments are escaped; an arrow belonging to the
  comment itself is written `\->`, so it cannot be mistaken for a step separator.
- **All four outputs** (`qwen.csv`, `sealion.csv`, `gemma.csv`, `suggestions.csv`)
  share seven columns: the five input columns followed by `candidate_label` and
  `valid`. They can each be read independently. `candidate_label` is the
  individual judge's label in the first three files and the consensus label in
  `suggestions.csv`; it is not a victim classifier prediction. `valid` is `YES`,
  `NO` or `UNSURE`. No separate meaning/mode scores, processing statuses,
  explanations or accept/reject columns are saved.

Gemma is called if either the candidate label or validity disagrees, or either
is `UNSURE`. Final label and validity each require two matching non-uncertain
votes; failure to reach that threshold stays `UNSURE`. Both failed responses and
genuine uncertainty are unresolved votes. No path is removed, including `valid=NO`.
These are **LLM-judged results for the current version**, without human validation. Do not label a whole variant
valid in every condition because one path was accepted. Rerunning the merge replaces `suggestions.csv` without rerunning model inference. With merged validity, these files do not identify
whether an invalid path failed on meaning or edit level.

Files are joined by **`source_id` + `path`**, including when judge row orders differ.
Source text and labels are also checked when reading saved votes. Source IDs link
back to the candidate CSVs; complete paths distinguish different routes to the
same variant when recovering classifier/mode/beam/Q/G provenance. No generated
IDs, revision/signature columns or metadata files are written.
Judges use greedy decoding, so there is no judge sampling-seed setting.

Every completed verdict is flushed to its judge CSV, including failures. Reruns
skip saved source/path pairs. To resume in a new
Kaggle session, restore those CSVs into `OUTPUT_DIR` alongside the same input CSV.
Changing prompts/settings requires a different output directory or explicitly
removing the old judge outputs. One process may write each judge CSV at a time.
Human verification and judge-model upgrades are deferred; they are not a required
phase in the current version. Final test-set export remains future work. If human
review is added later, keep those labels separate from the existing LLM votes.

Model interfaces follow the official cards:
[Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-9B),
[SEA-LION](https://huggingface.co/aisingapore/Llama-SEA-LION-v3.5-8B-R),
and [Gemma 4](https://huggingface.co/google/gemma-4-12B-it).


Current pilot consensus: **172/1,773 paths (9.70%)** receive `valid=YES`, representing
**141 distinct source/variant pairs**. Gemma judged 925 routed paths. SEA-LION
returned `NO` on every case, so all final `YES` results require agreement between
Qwen and Gemma. There are 495 unresolved candidate labels, all with final
`valid=NO`. Path acceptance is distinct from source-based ASR_LLM; these results
do not establish judge accuracy. CSVs remain under `artifacts/judges/`.
