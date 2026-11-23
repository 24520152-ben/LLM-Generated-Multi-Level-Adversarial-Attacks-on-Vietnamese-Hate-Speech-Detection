# ViHSD Baselines

## Run on Kaggle

Start a Kaggle session with GPU and Internet enabled. Accept the access conditions
for [ViHSD](https://huggingface.co/datasets/uitnlp/vihsd) and attach a Hugging Face
read token to the notebook as a Kaggle secret named `HF_TOKEN`.

Push the project source to GitHub, then open `scripts/train.ipynb` in VS Code.
Select the Kaggle kernel and click **Run All**. The notebook clones
`GITHUB_REPO_URL` into `/kaggle/working/ie403` on the first run and reuses that
checkout for later runs. Use a fresh Kaggle session to clone a newer revision.

ViSoBERT runs by default. To try one epoch, change the first cell to use these
training options.

```python
BASELINE_OPTIONS = {"model": "visobert", "epochs": 1}
```

Omit `"epochs": 1` to train for the default 10 epochs and select the checkpoint
with the highest validation macro-F1.

## Run outputs

Each run under `artifacts/baselines/<model>/<run_id>/` saves two items.

- `checkpoint/` contains the model and the files needed by `load_classifier()`.
- `metrics.csv` has four columns (`split`, `epoch`, `macro_f1`, `accuracy`). A
  Transformer run writes one validation row per epoch and one test row for the
  selected epoch. SVM writes one row per split with an empty `epoch` column.

Baseline defaults are in `src/vihsd_attack/config.py`. Run-specific overrides
are in the first cell of `scripts/train.ipynb`.

To regenerate per-example predictions, load `checkpoint/` with
`load_classifier()` and call `predict()` on texts from the original dataset
snapshot and split.
