# fMRI sequence models ported to the ml4fmri external-model API

Drop-in model files extracted from two projects and adapted to the
`cvbench(..., custom_models=...)` interface described in
`ml4fmri/scripts/ml4fmri_tutorial.ipynb`, section 1.1.

| File | Classes | Origin |
|---|---|---|
| `MDD_LSTM.py` | `MDDLSTM`, `MDDLSTMAttention`, `AttentionPooling` | `MDD_preproc/mdd_classifier/models/lstm.py`, `attention.py` |
| `MDD_GRU.py` | `MDDGRU`, `MDDGRUAttention`, `AttentionPooling` | `MDD_preproc/mdd_classifier/models/gru.py`, `attention.py` |
| `MDD_Transformer.py` | `MDDTransformer` | `MDD_preproc/.../transformer.py` + `TimeSeriesTransformerEncoder` from `multimodal_transformer.py` |
| `phase_LSTM.py` | `PhaseBiLSTM`, `PhaseBiLSTMAttention`, `with_seq_len` | `phase_prediction/models/bilstm.py`, `bisltm_attention.py` |
| `phase_GRU.py` | `PhaseBiGRU`, `PhaseBiGRUAttention`, `with_seq_len` | `phase_prediction/models/gru.py`, `bigru_attention.py` |
| `phase_Transformer.py` | `PhaseTransformer`, `with_seq_len` | `phase_prediction/models/transformer.py` |

Every file is self-contained — only `torch` and `ml4fmri.models.helper_functions` — so it can be
copied anywhere on the path.

## Usage

```python
from ml4fmri import cvbench
from MDD_LSTM import MDDLSTM, MDDLSTMAttention
from MDD_GRU import MDDGRU, MDDGRUAttention
from MDD_Transformer import MDDTransformer
from phase_LSTM import PhaseBiLSTM, PhaseBiLSTMAttention, with_seq_len
from phase_GRU import PhaseBiGRU, PhaseBiGRUAttention
from phase_Transformer import PhaseTransformer

T = DATA.shape[1]        # number of TRs

report = cvbench(
    DATA, LABELS,                      # (subjects, time, components), (subjects,)
    models=["meanMLP", "LSTM"],        # built-in baselines
    custom_models=[
        MDDLSTM, MDDLSTMAttention, MDDGRU, MDDGRUAttention, MDDTransformer,
        with_seq_len(PhaseBiLSTM, T),
        with_seq_len(PhaseBiLSTMAttention, T),
        with_seq_len(PhaseBiGRU, T),
        with_seq_len(PhaseBiGRUAttention, T),
        with_seq_len(PhaseTransformer, T),
    ],
    n_folds=5,
)
report.plot_scores()
```

Class names are prefixed so they run *alongside* ml4fmri's built-in `LSTM` and `Transformer`
rather than overriding them. Two classes sharing a name silently override each other, so pass a
`name=` to `with_seq_len` if you want to benchmark two configurations of the same class in one
run.

## The sequence-length problem (phase_ models only)

`cvbench` constructs a custom model as `Model(input_size=D, output_size=C)` — no time length.
The phase_prediction models need it: they flatten every time step into the FC head
(`Linear(seq_len * hidden * n_directions, hidden)`) and normalise across the time axis with
`BatchNorm1d(seq_len)`. Two supported ways out:

1. **`with_seq_len(cls, T, name=None, **fixed)`** — returns a subclass with `seq_len` (and any
   other hyperparameters) bound, so it still takes only the two required arguments. This is the
   faithful path and the recommended one. The helper is defined in each `phase_*.py`.

2. **Leave `seq_len=None`** (the default) and the model becomes length-agnostic: the flatten head
   is replaced by mean pooling and the time-axis BatchNorm by a LayerNorm over features. This
   runs on any input but is a real architectural deviation — it drops the per-timestep weights
   that are arguably the point of the original design. The `pool` and `norm` arguments override
   the automatic choice.

The MDD_ models pool over time and need no length at all.

## API added to each model

Required by `cvbench`, following `ml4fmri/src/ml4fmri/models/LSTM.py`:

- `__init__(input_size, output_size, ...)` — only these two required; everything else is a
  defaulted keyword hyperparameter.
- `forward(x) -> (logits, {"logits": ..., "embedding": ...})`
- `compute_loss`, `handle_batch`, `get_optimizer`, `train_model` — thin wrappers over
  `basic_ce_loss`, `basic_handle_batch`, `basic_Adam_optimizer`, `BasicTrainer`.
- `prepare_dataloader(data, labels, batch_size, shuffle)` — `basic_dataloader(type="TS")`,
  z-scores along time. All six are time-series models.
- `self.lr` — reference learning rate: `3e-4` for the MDD_ models (their time-series runs),
  `1e-4` / `1e-6` for the phase_ models (see the table below).

## Deviations from the originals

**All models**

1. **Multi-class head.** Originals emitted one logit for BCE / MSE. Here the head emits
   `output_size` logits and training uses cross-entropy. Binary is `output_size=2`.
2. **`forward` signature.** Now always `(logits, loss_load_dict)`.
3. **No output softmax.** The phase_prediction models' `do_softmax` flag is dropped; ml4fmri
   applies softmax itself and expects raw logits.

**MDD_ models**

4. **RNN pooling default changed to `pool="last_bidir"`.** The originals summarise with
   `h_n[-1]`; for a bidirectional RNN `h_n` is `(num_layers * num_directions, B, H)` with
   directions interleaved last, so `h_n[-1]` is the *backward* direction of the top layer only —
   the forward pass is discarded. `pool="last"` reproduces that original behaviour if you want
   to compare.
5. **Transformer consolidation.** `TransformerEncoderModel` and `TimeSeriesTransformerEncoder`
   were near-duplicates; merged into `MDDTransformer` with `pool` / `input_dropout` /
   `fc_hidden_dims` switches. `MDDTransformer.from_modality_comparison(D, C)` reproduces the
   exact config run by `experiments_clf/modality_comparison.sh` (d_model=384, nhead=8, 4 layers,
   ffn=768, CLS pooling, no MLP head).

**phase_ models**

6. **`seq_len` handling** — see above.
7. **Defaults come from phase_prediction's own grid search**, not the source files' constructor
   defaults — see the table below.
8. **`PhaseTransformer` head count is constrained.** The encoder runs at `d_model = input_size`
   (no input projection, faithful to the original), so `nhead` must divide `input_size` — with
   53 components that is 1 or 53; with 105 it is 1, 3, 5, 7, 15, 21, 35, 105. The constructor
   raises a clear error listing the valid divisors instead of failing inside torch.
9. **`PhaseTransformer`'s unused `bidirectional` argument** is dropped.

## phase_ defaults and where they come from

`experiments/C001_grid_mddd.sh` generates a grid (8 MDDD datasets x 3 models x 4 model_kwargs x
3 batch sizes x 3 LRs x 5 folds) into `grids/C001_grid.txt`, run as a slurm array. Results are
summarised in `tables/C001_valid_auc.csv` (and `B001_valid_auc.csv` for the earlier grid) as
max-over-epochs validation AUC averaged over folds. The defaults below are the best credible
configuration from that grid.

| Model | hidden_size | num_layers | lr | batch | source row |
|---|---|---|---|---|---|
| `PhaseBiLSTM` | 512 | 1 | 1e-4 | 128 | `ica53_timeseries`, val AUC 0.687 +/- 0.080 |
| `PhaseBiGRU` | 64 | 1 | 1e-4 | 128 | `ica53_timeseries`, val AUC 0.706 +/- 0.105 (best of any model on timecourses) |
| `PhaseTransformer` | 256 | 1 | 1e-6 | 128 | `dfnc_60s`, val AUC 0.786 +/- 0.076 — see note |

Batch size is set through `prepare_dataloader`'s default, since `cvbench` calls it without one.

Notes on those numbers:

- **The grid never tried an LR above 1e-4 on MDDD** (`{1e-8, 1e-6, 1e-4}`; B001 also tried 1e-2
  and it was clearly worst). The earlier 1e-3 default in these files was outside the swept range.
- **The grid ran `do_softmax=1` together with CrossEntropyLoss** — softmax applied twice. This
  port feeds raw logits to CE, so the tuned LRs are a starting point, not a transferable optimum.
- **Fold std is 0.03-0.15 and the statistic is max-over-epochs**, which is optimistically biased.
  Most differences between neighbouring settings sit inside the noise. In particular
  `PhaseBiLSTM` at `hidden_size=64` scored 0.628 vs 0.687 at 512 — statistically a wash, but 512
  with `pool="flatten"` puts ~120M weights in the FC head at 230 TRs. Consider overriding:
  `with_seq_len(PhaseBiLSTM, T, hidden_size=64)`.
- **`PhaseTransformer` has zero `ica53_timeseries` rows** in the C001 results while every other
  dataset has ~300 — those runs all crashed, almost certainly because the source default
  `nhead=2` cannot divide 53. It was never successfully trained on 53-component timecourses, so
  its defaults are borrowed from the other MDDD modalities.
- `num_layers=1` throughout: C001 only ran 1-layer configs, and B001 (which swept 1/3/5/10)
  found no meaningful difference, so 1 is the cheapest.

- `PhaseBiGRU`/`PhaseBiGRUAttention` skip their FC BatchNorm when a training batch contains a
  single sample (`n_train % batch_size == 1`), which would otherwise raise
  "Expected more than 1 value per channel". The time-axis BatchNorm is unaffected — it still has
  `n_components` values per channel at batch size 1.

## Caveats

- **No positional encoding** in either transformer, faithful to all three source classes. With
  mean or CLS pooling that makes them permutation-invariant over time — they read the timecourse
  as a set of TRs. `PhaseTransformer` with `pool="flatten"` does carry position through the head.
- `basic_dataloader` stacks all subjects into one tensor, so they must share a time length.
  MDD_preproc's own loader padded variable-length series per batch; that does not carry over.

## Verification

All nine classes were run end-to-end through `cvbench` on synthetic data
(60 subjects, 40 TRs, 53 components), 2 folds, CPU — forward, backward, early stopping, metric
logging and reporting all work, in both the pinned-`seq_len` and length-agnostic paths. Each
file also has a `__main__` shape / parameter-count check: `python phase_LSTM.py`.
