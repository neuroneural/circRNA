# pylint: disable=invalid-name, missing-function-docstring
"""
Transformer extracted from the phase_prediction project, adapted to the ml4fmri
external-model API.

Source
------
phase_prediction/models/transformer.py  (MyTransformer)

Different from the MDD_preproc transformer in two ways that matter:

* **No input projection.** The encoder runs at `d_model = input_size`, i.e.
  directly on the raw component vectors. Consequence: `nhead` must divide
  `input_size`. With 53 Neuromark components the only valid values are 1 and 53;
  with 105 they are 1, 3, 5, 7, 15, 21, 35, 105. The constructor checks this and
  raises with the list of valid divisors rather than failing deep inside torch.
* **Flatten head.** The sequence is summarised by flattening every time step
  (`Linear(seq_len * input_size, hidden_size)`), not pooled.

The sequence-length problem
---------------------------
ml4fmri instantiates a custom model as `Model(input_size=D, output_size=C)` and
never passes the time length, but the flatten head needs it. Two ways out, both
supported:

1. Pin it with the `with_seq_len` helper (faithful to the original):

       from phase_Transformer import PhaseTransformer, with_seq_len
       Model = with_seq_len(PhaseTransformer, DATA.shape[1])   # e.g. 140 TRs
       cvbench(DATA, LABELS, custom_models=Model, n_folds=5)

   `with_seq_len` returns a subclass with `seq_len` bound, so it still takes
   only `input_size`/`output_size`.

2. Leave `seq_len=None` (the default) and the model mean-pools over time
   instead of flattening — length-agnostic, but an architectural deviation.
   Set `pool` explicitly to override the automatic choice.

Other adaptations for ml4fmri
-----------------------------
* `seqlen`/`dim`/`num_classes` renamed to `seq_len`/`input_size`/`output_size`.
* `output_size` is the number of classes and training uses cross-entropy; the
  original emitted a single logit.
* The original's unused `bidirectional` argument and its optional `nn.Softmax`
  output (`do_softmax`) are dropped — ml4fmri expects raw logits.
* `forward` returns `(logits, loss_load_dict)`.
* Added `lr`, `compute_loss`, `handle_batch`, `get_optimizer`,
  `prepare_dataloader` and `train_model`.

Note: no positional encoding, faithful to the original. With `pool="mean"` the
model is therefore permutation-invariant over time; with `pool="flatten"` the
head does carry position information.
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


def _divisors(n: int):
    return [d for d in range(1, n + 1) if n % d == 0]


class PhaseTransformer(nn.Module):
    """
    TIME SERIES MODEL

    phase_prediction's MyTransformer: a transformer encoder applied directly to
    the raw feature vectors (d_model = input_size, no projection), then a
    flatten-and-FC head.
    Expected input shape: [batch_size, time_length, input_feature_size].
    Output: [batch_size, n_classes]
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        seq_len: Optional[int] = None,
        hidden_size: int = 1024,
        nhead: int = 1,
        num_layers: int = 3,
        dropout: float = 0.1,
        pool: Literal["auto", "flatten", "mean"] = "auto",
        lr: float = 1e-3,
    ):
        """
        Initialize the model.
        Args:
            input_size (int): Size of the vector at each time step (`dim` in the original). \
                Also the transformer's d_model, since there is no input projection. \
                Common to all models.
            output_size (int): Number of classes for classification. Common to all models.
            seq_len (int, optional): Number of time points (`seqlen` in the original). \
                Required for pool="flatten". If None, the model mean-pools over time. \
                Use `with_seq_len` to pin it.
            hidden_size (int, hyperparameter): Width of the FC head. Defaults to 1024.
            nhead (int, hyperparameter): Attention heads. Must divide `input_size`. Defaults to 1.
            num_layers (int, hyperparameter): Encoder layers. Defaults to 3.
            dropout (float, hyperparameter): Dropout in the encoder and before the output layer. \
                Defaults to 0.1.
            pool (str, hyperparameter): "flatten" (original) or "mean". "auto" picks "flatten" \
                when seq_len is given, else "mean".
            lr (float, hyperparameter): Reference learning rate. Defaults to 1e-3.
        """
        super().__init__()
        self.lr = lr

        if input_size % nhead != 0:
            raise ValueError(
                f"nhead={nhead} does not divide input_size={input_size}. This model runs the "
                f"encoder at d_model=input_size, so nhead must be one of {_divisors(input_size)}."
            )

        if pool == "auto":
            pool = "flatten" if seq_len is not None else "mean"
        if pool == "flatten" and seq_len is None:
            raise ValueError(
                "pool='flatten' needs seq_len; pin it with with_seq_len(cls, DATA.shape[1])"
            )
        self.pool = pool
        self.seq_len = seq_len

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        fc_in = seq_len * input_size if pool == "flatten" else input_size
        self.linear = nn.Linear(fc_in, hidden_size)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.outlayer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): (batch, seq_len, input_size).
        Returns:
            (logits, loss_load): logits (batch, n_classes) and a dict with \
                "logits" and "embedding".
        """
        h = self.transformer_encoder(x)
        pooled = torch.flatten(h, 1) if self.pool == "flatten" else h.mean(dim=1)
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
    def prepare_dataloader(data, labels, batch_size: int = 64, shuffle: bool = True):
        """
        Returns a torch DataLoader producing batches appropriate for the model.
        Args:
            data (array-like): Time series data of shape (B, T, D).
            labels (array-like): Class labels for the data.
            batch_size (int, optional): Batch size. Defaults to 64.
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


if __name__ == "__main__":
    B, T, D, C = 8, 140, 53, 2
    x = torch.randn(B, T, D)
    variants = {
        "PhaseTransformer (seq_len pinned, flatten)": with_seq_len(PhaseTransformer, T)(D, C),
        "PhaseTransformer (length-agnostic, mean)": PhaseTransformer(D, C),
        "PhaseTransformer (nhead=53)": PhaseTransformer(D, C, nhead=53),
    }
    for name, model in variants.items():
        logits, load = model(x)
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{name}: logits {tuple(logits.shape)}, params {n:,}")
    try:
        PhaseTransformer(D, C, nhead=4)
    except ValueError as err:
        print(f"nhead=4 correctly rejected: {err}")
