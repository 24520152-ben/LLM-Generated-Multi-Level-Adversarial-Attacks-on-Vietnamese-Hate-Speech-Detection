# ViHSD Baselines

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
Each model has one run with seed 42; confidence intervals have not been estimated.

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
