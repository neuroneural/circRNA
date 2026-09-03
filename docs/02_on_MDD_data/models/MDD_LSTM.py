# pylint: disable=invalid-name, missing-function-docstring
"""
LSTM models extracted from the MDD_preproc project, adapted to the ml4fmri
external-model API.

Source
------
MDD_preproc/mdd_classifier/models/lstm.py      (LSTMModel, LSTMAttentionModel)
MDD_preproc/mdd_classifier/models/attention.py (AttentionPooling)

In MDD_preproc these were driven by `training/unimodal_trainer.py`, which fed
ICA component time courses (B, T, n_components) with the configuration
hidden_size=128, num_layers=2, bidirectional=True, dropout=0.1,
fc_hidden_dims=[64] (the GRU sibling was the one wired into the shell scripts;
the LSTM was implemented but never launched).

Adaptations for ml4fmri
-----------------------
* `input_dim`/`output_dim` renamed to the required `input_size`/`output_size`,
  and are now the only positional-free required arguments.
* `output_size` is the number of classes: the head emits `output_size` logits
  and training uses cross-entropy, instead of the single-logit BCE head used in
  MDD_preproc.
* `forward` returns `(logits, loss_load_dict)` as ml4fmri expects, rather than
  a bare tensor / `(out, embedding)` pair. The embedding is still available in
  the dict under "embedding".
* Added `lr`, `compute_loss`, `handle_batch`, `get_optimizer`,
  `prepare_dataloader` and `train_model`.

Note on RNN pooling
-------------------
MDD_preproc takes `h_n[-1]` as the sequence summary. For a bidirectional LSTM
`h_n` is (num_layers * num_directions, B, H) with directions interleaved last,
so `h_n[-1]` is the *backward* direction of the top layer only — the forward
direction is dropped. That is almost certainly unintended, so the default here
is `pool="last_bidir"`, which concatenates both top-layer directions in the
conventional way. Pass `pool="last"` to reproduce the original behaviour
exactly, or `pool="attention"` / `MDDLSTMAttention` for attention pooling.

Usage
-----
    from ml4fmri import cvbench
    from MDD_LSTM import MDDLSTM, MDDLSTMAttention

    report = cvbench(DATA, LABELS,
                     models=["meanMLP"],
                     custom_models=[MDDLSTM, MDDLSTMAttention],
                     n_folds=5)

Rename a class to `LSTM` if you want it to override ml4fmri's built-in LSTM
rather than run alongside it.
"""

from typing import Literal, Sequence

import torch
from torch import nn

from ml4fmri.models.helper_functions import (
    BasicTrainer,
    basic_Adam_optimizer,
    basic_ce_loss,
    basic_dataloader,
    basic_handle_batch,
)


class AttentionPooling(nn.Module):
    """Simple attention pooling over sequences (MDD_preproc models/attention.py)."""

    def __init__(self, input_dim: int, attention_type: Literal["additive", "dot"] = "additive"):
        """
        Args:
            input_dim (int): Feature dimension per time step.
            attention_type (str): "additive" or "dot".
        """
        super().__init__()
        self.attention_type = attention_type
        if attention_type == "additive":
            self.query = nn.Parameter(torch.zeros(input_dim))
            self.proj = nn.Linear(input_dim, input_dim)
        elif attention_type == "dot":
            self.query = nn.Parameter(torch.zeros(input_dim))
            self.proj = None
        else:
            raise ValueError("attention_type must be 'additive' or 'dot'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (batch, seq_len, input_dim).
        Returns:
            torch.Tensor: (batch, input_dim) pooled embeddings.
        """
        if self.attention_type == "additive":
            scores = torch.tanh(self.proj(x)) @ self.query
        else:
            scores = x @ self.query
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return torch.sum(weights * x, dim=1)


class MDDLSTM(nn.Module):
    """
    TIME SERIES MODEL

    Configurable LSTM with an optional MLP head, from MDD_preproc.
    Expected input shape: [batch_size, time_length, input_feature_size].
    Output: [batch_size, n_classes]
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1,
        fc_hidden_dims: Sequence[int] = (64,),
        pool: Literal["last_bidir", "last", "attention"] = "last_bidir",
        attention_type: Literal["additive", "dot"] = "additive",
        lr: float = 3e-4,
    ):
        """
        Initialize the LSTM model.
        Args:
            input_size (int): Size of the vector at each time step in the input time series. \
                Common to all models, for FNC models it is the number of nodes in the FNC matrix.
            output_size (int): Number of classes for classification. Common to all models.
            hidden_size (int, hyperparameter): LSTM hidden size. Defaults to 128.
            num_layers (int, hyperparameter): Number of LSTM layers. Defaults to 2.
            bidirectional (bool, hyperparameter): Whether to use a bidirectional LSTM. Defaults to True.
            dropout (float, hyperparameter): Dropout between LSTM layers (ignored if num_layers == 1). Defaults to 0.1.
            fc_hidden_dims (Sequence[int], hyperparameter): Hidden dims of the FC head. Defaults to (64,).
            pool (str, hyperparameter): Sequence pooling. Defaults to "last_bidir", which concatenates \
                both top-layer directions. "last" reproduces MDD_preproc's `h_n[-1]` (backward direction \
                only when bidirectional); "attention" uses attention pooling.
            attention_type (str, hyperparameter): "additive" or "dot", used when pool="attention".
            lr (float, hyperparameter): Reference learning rate. Defaults to 3e-4 \
                (the LR used for the time-series runs in MDD_preproc).
        """
        super().__init__()
        self.lr = lr

        if pool not in ("last_bidir", "last", "attention"):
            raise ValueError("pool must be 'last_bidir', 'last' or 'attention'")
        self.pool = pool
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        out_dim = hidden_size * (2 if bidirectional else 1)
        # pool="last" keeps a single direction, matching MDD_preproc's original h_n[-1]
        pooled_dim = hidden_size if pool == "last" else out_dim

        self.attention = (
            AttentionPooling(out_dim, attention_type=attention_type)
            if pool == "attention"
            else None
        )

        head_layers = []
        prev_dim = pooled_dim
        for hidden_dim in fc_hidden_dims:
            head_layers.append(nn.Linear(prev_dim, hidden_dim))
            head_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        self.head = nn.Sequential(*head_layers)
        self.out = nn.Linear(prev_dim, output_size)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): (batch, seq_len, input_size).
        Returns:
            (logits, loss_load): logits of shape (batch, n_classes) and a dict \
                with "logits" and "embedding" for loss computation / inspection.
        """
        outputs, (h_n, _) = self.lstm(x)

        if self.pool == "attention":
            pooled = self.attention(outputs)
        elif self.pool == "last_bidir":
            pooled = outputs[:, -1, : self.hidden_size]
            if self.bidirectional:
                pooled = torch.cat((pooled, outputs[:, 0, self.hidden_size :]), dim=1)
        else:  # "last" — faithful to MDD_preproc
            pooled = h_n[-1]

        embedding = self.head(pooled) if len(self.head) > 0 else pooled
        logits = self.out(embedding)

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


class MDDLSTMAttention(MDDLSTM):
    """
    TIME SERIES MODEL

    MDD_preproc's LSTMAttentionModel: same LSTM, but pooled with attention over
    all sequence outputs instead of the final hidden state.
    Expected input shape: [batch_size, time_length, input_feature_size].
    Output: [batch_size, n_classes]
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1,
        fc_hidden_dims: Sequence[int] = (64,),
        attention_type: Literal["additive", "dot"] = "additive",
        lr: float = 3e-4,
    ):
        """Same arguments as MDDLSTM, with `pool` fixed to "attention"."""
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            fc_hidden_dims=fc_hidden_dims,
            pool="attention",
            attention_type=attention_type,
            lr=lr,
        )


if __name__ == "__main__":
    B, T, D, C = 8, 120, 53, 2
    x = torch.randn(B, T, D)
    for cls in (MDDLSTM, MDDLSTMAttention):
        model = cls(input_size=D, output_size=C)
        logits, load = model(x)
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{cls.__name__}: logits {tuple(logits.shape)}, "
              f"embedding {tuple(load['embedding'].shape)}, params {n}")
