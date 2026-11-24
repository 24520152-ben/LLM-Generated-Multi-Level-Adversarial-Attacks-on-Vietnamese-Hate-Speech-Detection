# ViHSD Baselines

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

The PhoBERT notebook runs `uv sync --locked` in the checkout and launches
training in that project environment. The first sync installs the locked
PyTorch build and PhoBERT dependencies. The other notebooks run in the selected
Kaggle kernel.

Transformer notebooks run for 10 epochs and select the checkpoint with the
highest validation macro-F1. Set `"epochs": 1` in the first cell for a test run.

To train two models concurrently on two T4 GPUs, open two notebooks with
separate Python kernels on the same Kaggle server. Set `"device": "cuda:0"`
in one notebook and `"device": "cuda:1"` in the other. Each training run
prints its selected device and keeps its own checkpoint and CSV output.

## Run outputs

Each run under `artifacts/baselines/<model>/<run_id>/` saves two items.

- `checkpoint/` contains the model and the files needed by `load_classifier()`.
- `metrics.csv` has four columns (`split`, `epoch`, `macro_f1`, `accuracy`). A
  Transformer run writes one validation row per epoch and one test row for the
  selected epoch. SVM writes one row per split with an empty `epoch` column.

Baseline defaults are in `src/vihsd_attack/config.py`. Run-specific overrides
are in the first cell of each notebook.

To regenerate per-example predictions, load `checkpoint/` with
`load_classifier()` and call `predict()` on texts from the original dataset
snapshot and split.
