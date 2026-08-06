# pylint: disable=invalid-name, missing-function-docstring
"""
LSTM models extracted from the phase_prediction project, adapted to the ml4fmri
external-model API.

Source
------
phase_prediction/models/bilstm.py            (BiLSTM)
phase_prediction/models/bisltm_attention.py  (BiLSTMAttention)

These are a different design from the MDD_preproc RNNs: they summarise the
sequence by *flattening every time step* into one big vector
(`Linear(seq_len * hidden * n_directions, hidden)`) rather than pooling, and
they normalise across the time axis with `BatchNorm1d(seq_len)`.

The sequence-length problem
---------------------------
ml4fmri instantiates a custom model as `Model(input_size=D, output_size=C)` and
never passes the time length, but the flatten head and the time-axis BatchNorm
both need it. Two ways out, both supported:

1. Pin it with the `with_seq_len` helper (faithful to the original):

       from phase_LSTM import PhaseBiLSTM, with_seq_len
       Model = with_seq_len(PhaseBiLSTM, DATA.shape[1])   # e.g. 140 TRs
       cvbench(DATA, LABELS, custom_models=Model, n_folds=5)

   `with_seq_len` returns a subclass with `seq_len` bound, so it still takes
   only `input_size`/`output_size`. It keeps the original class name in reports
   unless you pass `name=`.

2. Leave `seq_len=None` (the default) and the model becomes length-agnostic:
   the flatten head is replaced by mean pooling over time and the time-axis
   BatchNorm by a LayerNorm over features. This is a real architectural
   deviation — it drops the per-timestep weights that are arguably the whole
   point of the original design — but it runs on any input and is a reasonable
   baseline. `pool` and `norm` can be set explicitly to override the automatic
   choice.

Other adaptations for ml4fmri
-----------------------------
* `seqlen`/`dim`/`num_classes` renamed to `seq_len`/`input_size`/`output_size`;
  `input_size` and `output_size` are the only required arguments.
* `output_size` is the number of classes: the head emits `output_size` logits
  and training uses cross-entropy. The originals emitted a single logit
  (regression / BCE style) and `BiLSTMAttention` had `num_classes` hard-coded
  to 1.
* The originals' optional `nn.Softmax` output (`do_softmax`) is dropped —
  ml4fmri applies softmax itself and expects raw logits.
* `forward` returns `(logits, loss_load_dict)`.
* Added `lr`, `compute_loss`, `handle_batch`, `get_optimizer`,
  `prepare_dataloader` and `train_model`.

Where the defaults come from
----------------------------
Taken from phase_prediction's own grid search on MDDD/Diagnosis
(`experiments/C001_grid_mddd.sh` -> `grids/C001_grid.txt`, summarised in
`tables/C001_valid_auc.csv`; `B001_valid_auc.csv` is the earlier MDDD grid).
The score is max-over-epochs validation AUC averaged over 5 folds.

Best `ica53_timeseries` config for `bilstm`, restricting to LRs above the
degenerate 1e-8: batch 128, lr 1e-4, hidden_size 512, num_layers 1 ->
val AUC 0.687 +/- 0.080. (The nominal overall best was lr=1e-8 at 0.696, which
is not a credible learning rate.) B001 swept num_layers 1/3/5/10 and found no
meaningful difference, so 1 is used as the cheapest.

Heads-up on size: hidden_size=512 bidirectional with pool="flatten" at 230 TRs
puts ~120M weights in the FC head alone. hidden_size=64 scored 0.628 on the
same grid -- well inside the fold noise -- for ~1/60th of the parameters.

Three caveats on those numbers:

* The grid ran `do_softmax=1` *together with* CrossEntropyLoss, i.e. softmax was
  applied twice. This port feeds raw logits to CE (correctly), so the tuned LRs
  are a starting point, not a transferable optimum.
* Fold std is 0.03-0.15 and the statistic is max-over-epochs, which is
  optimistically biased. Most differences between neighbouring settings are
  inside the noise.
* The grid never went above lr=1e-4 on MDDD (`{1e-8, 1e-6, 1e-4}`); 1e-2 was
  tried only in B001 and was clearly worst.

Note: with `pool="flatten"` these models are large -- see the size heads-up
above before raising `hidden_size` or the number of TRs.
"""

from typing import Literal, Optional

import torch
from torch import nn

from ml4fmri.models.helper_functions import (
    BasicTrainer,
    basic_Adam_optimizer,
    basic_ce_loss,
    basic_dataloader,
    basic_handle_batch,
)


def with_seq_len(cls, seq_len: int, name: Optional[str] = None, **fixed):
    """
    Bind `seq_len` (and any other hyperparameters) so the class satisfies
    ml4fmri's two-argument constructor contract.

    Args:
        cls (type): One of the Phase* model classes in this module.
        seq_len (int): Number of time points in your data, i.e. DATA.shape[1].
        name (str, optional): Class name to show in cvbench reports. \
            Defaults to the original class name.
        **fixed: Any other constructor kwargs to pin.

    Returns:
        type: A subclass taking only `input_size` and `output_size`.
    """
    defaults = dict(fixed)
    defaults["seq_len"] = seq_len

    def __init__(self, input_size: int, output_size: int, **kwargs):
        merged = {**defaults, **kwargs}
        cls.__init__(self, input_size=input_size, output_size=output_size, **merged)

    return type(name or cls.__name__, (cls,), {"__init__": __init__,
                                               "__doc__": cls.__doc__})


class PhaseBiLSTM(nn.Module):
    """
    TIME SERIES MODEL

    phase_prediction's BiLSTM: bidirectional LSTM, BatchNorm over the time axis,
    then the whole sequence flattened into a single FC head.
    Expected input shape: [batch_size, time_length, input_feature_size].
    Output: [batch_size, n_classes]
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        seq_len: Optional[int] = None,
        hidden_size: int = 512,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.1,
        pool: Literal["auto", "flatten", "mean", "last_bidir", "attention"] = "auto",
        norm: Literal["auto", "batch", "layer", "none"] = "auto",
        activation: Literal["relu", "tanh", "lrelu"] = "relu",
        activation_alpha: float = 0.0,
        lr: float = 1e-4,
    ):
        """
        Initialize the model.
        Args:
            input_size (int): Size of the vector at each time step (`dim` in the original). \
                Common to all models.
            output_size (int): Number of classes for classification. Common to all models.
            seq_len (int, optional): Number of time points (`seqlen` in the original). \
                Required for pool="flatten" and norm="batch". If None, the model falls back \
                to length-agnostic mean pooling and LayerNorm. Use `with_seq_len` to pin it.
            hidden_size (int, hyperparameter): LSTM hidden size, and the width of the FC head. \
                Defaults to 512, the grid-best for bilstm on ICA-53 timecourses. See the size \
                warning in the module docstring; 64 is a far cheaper near-equivalent.
            num_layers (int, hyperparameter): Number of LSTM layers. Defaults to 1 (grid-best).
            bidirectional (bool, hyperparameter): Bidirectional LSTM. Defaults to True.
            dropout (float, hyperparameter): Dropout before the output layer (`drp`). Defaults to 0.1.
            pool (str, hyperparameter): "flatten" (original), "mean", "last_bidir" or "attention". \
                "auto" picks "flatten" when seq_len is given, else "mean".
            norm (str, hyperparameter): "batch" (original, BatchNorm1d over the time axis), \
                "layer", or "none". "auto" picks "batch" when seq_len is given, else "layer".
            activation (str, hyperparameter): "relu", "tanh" or "lrelu". Defaults to "relu".
            activation_alpha (float, hyperparameter): Negative slope for "lrelu". Defaults to 0.0.
            lr (float, hyperparameter): Reference learning rate. Defaults to 1e-4, the best \
                credible value in phase_prediction's MDDD grid.
        """
        super().__init__()
        self.lr = lr

        if pool == "auto":
            pool = "flatten" if seq_len is not None else "mean"
        if norm == "auto":
            norm = "batch" if seq_len is not None else "layer"
        if pool == "flatten" and seq_len is None:
            raise ValueError(
                "pool='flatten' needs seq_len; pin it with with_seq_len(cls, DATA.shape[1])"
            )
        if norm == "batch" and seq_len is None:
            raise ValueError("norm='batch' needs seq_len; use norm='layer' or pin seq_len")

        self.pool = pool
        self.norm_kind = norm
        self.seq_len = seq_len
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        mul = 2 if bidirectional else 1
        self.rnn = self._make_rnn(input_size, hidden_size, num_layers, bidirectional)

        if norm == "batch":
            self.norm = nn.BatchNorm1d(seq_len)      # normalises across the time axis
        elif norm == "layer":
            self.norm = nn.LayerNorm(hidden_size * mul)
        else:
            self.norm = nn.Identity()

        if pool == "flatten":
            fc_in = seq_len * hidden_size * mul
        elif pool == "last_bidir":
            fc_in = hidden_size * mul
        else:                                        # "mean" or "attention"
            fc_in = hidden_size * mul

        self.attention = nn.Linear(hidden_size * mul, 1) if pool == "attention" else None

        self.linear = nn.Linear(fc_in, hidden_size)
        if activation == "tanh":
            self.act = nn.Tanh()
        elif activation == "lrelu":
            self.act = nn.LeakyReLU(negative_slope=activation_alpha)
        else:
            self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.outlayer = nn.Linear(hidden_size, output_size)

    @staticmethod
    def _make_rnn(input_size, hidden_size, num_layers, bidirectional):
        return nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )

    def _pool(self, h):
        """Reduce (B, T, H*mul) to (B, fc_in)."""
        if self.pool == "flatten":
            return torch.flatten(h, 1)
        if self.pool == "mean":
            return h.mean(dim=1)
        if self.pool == "last_bidir":
            out = h[:, -1, : self.hidden_size]
            if self.bidirectional:
                out = torch.cat((out, h[:, 0, self.hidden_size :]), dim=1)
            return out
        weights = torch.softmax(self.attention(h).squeeze(-1), dim=-1)
        return torch.sum(h * weights.unsqueeze(-1), dim=1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): (batch, seq_len, input_size).
        Returns:
            (logits, loss_load): logits (batch, n_classes) and a dict with \
                "logits" and "embedding".
        """
        h, _ = self.rnn(x)
        h = self.norm(h)
        pooled = self._pool(h)
        embedding = self.act(self.linear(pooled))
        logits = self.outlayer(self.dropout(embedding))
        return logits, {"logits": logits, "embedding": embedding}

    #### Helper functions for model training and evaluation ####

    def compute_loss(self, loss_load, targets):
        """
        Standard loss computation routine for models.
        Args:
            loss_load (dict): Forward's second output for the batch.
            targets (torch.Tensor): True labels for the batch.
        Returns
        -------
        loss : Tensor
            Loss for backpropagation.
        logs : dict
            Dictionary containing the loss components for logs.
        """
        loss, loss_log = basic_ce_loss(loss_load["logits"], targets)
        return loss, loss_log

    def handle_batch(self, batch):
        """
        Standard batch handling routine for models.
        Args:
            batch (tuple): A batch containing the time series data and labels as a tuple.
        Returns
        -------
        loss : Tensor
            Loss for backpropagation.
        batch_log : dict
            Dictionary of classification metrics and losses for logs.
        """
        loss, batch_log = basic_handle_batch(self, batch)
        return loss, batch_log

    def get_optimizer(self, lr=None):
        """Standard optimizer getter routine for models."""
        if lr is None:
            lr = self.lr
        return basic_Adam_optimizer(self, lr)

    @staticmethod
    def prepare_dataloader(data, labels, batch_size: int = 128, shuffle: bool = True):
        """
        Returns a torch DataLoader producing batches appropriate for the model.
        Args:
            data (array-like): Time series data of shape (B, T, D).
            labels (array-like): Class labels for the data.
            batch_size (int, optional): Batch size. Defaults to 128 — cvbench calls this \
                without a batch size, and 128 beat 64 and 256 in phase_prediction's grid.
            shuffle (bool, optional): Whether to shuffle batching. Defaults to True.
        Returns:
            DataLoader: batches of time series data and labels.
        """
        return basic_dataloader(data, labels, type="TS", batch_size=batch_size, shuffle=shuffle)

    def train_model(
        self,
        train_loader,
        val_loader,
        test_loader,
        epochs: int = 200,
        lr: float = None,
        device: str = None,
        patience: int = 30,
    ):
        """
        Standard model training routine.
        Args:
            train_loader (DataLoader): Training set loader.
            val_loader (DataLoader): Validation set loader, used for early stopping.
            test_loader (DataLoader): Test set loader.
            epochs (int, optional): Max training epochs. Defaults to 200.
            lr (float, optional): Optimizer LR (default: model's self.lr).
            device (str, optional): "cuda", "mps" or "cpu". Default: auto-detect.
            patience (int, optional): Early stopping patience in epochs. Defaults to 30.
        Returns
        -------
        (train_log, test_log, predictions_log) : tuple of pandas.DataFrame
        """
        trainer = BasicTrainer(
            model=self,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            epochs=epochs,
            lr=lr,
            device=device,
            patience=patience,
        )
        return trainer.run()


class PhaseBiLSTMAttention(PhaseBiLSTM):
    """
    TIME SERIES MODEL

    phase_prediction's BiLSTMAttention: same stack, but the sequence is reduced
    by a learned additive attention weighting instead of being flattened. This
    variant is length-agnostic, so `seq_len` is only needed for norm="batch".
    Expected input shape: [batch_size, time_length, input_feature_size].
    Output: [batch_size, n_classes]
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        seq_len: Optional[int] = None,
        hidden_size: int = 512,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.1,
        norm: Literal["auto", "batch", "layer", "none"] = "auto",
        activation: Literal["relu", "tanh", "lrelu"] = "relu",
        activation_alpha: float = 0.0,
        lr: float = 1e-4,
    ):
        """Same arguments as PhaseBiLSTM, with `pool` fixed to "attention"."""
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            seq_len=seq_len,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            pool="attention",
            norm=norm,
            activation=activation,
            activation_alpha=activation_alpha,
            lr=lr,
        )


if __name__ == "__main__":
    B, T, D, C = 8, 140, 53, 2
    x = torch.randn(B, T, D)
    variants = {
        "PhaseBiLSTM (seq_len pinned, flatten+batchnorm)": with_seq_len(PhaseBiLSTM, T)(D, C),
        "PhaseBiLSTM (length-agnostic, mean+layernorm)": PhaseBiLSTM(D, C),
        "PhaseBiLSTMAttention (seq_len pinned)": with_seq_len(PhaseBiLSTMAttention, T)(D, C),
        "PhaseBiLSTMAttention (length-agnostic)": PhaseBiLSTMAttention(D, C),
    }
    for name, model in variants.items():
        logits, load = model(x)
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{name}: logits {tuple(logits.shape)}, params {n:,}")
