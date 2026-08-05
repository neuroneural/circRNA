# pylint: disable=invalid-name, missing-function-docstring
"""
Transformer encoder extracted from the MDD_preproc project, adapted to the
ml4fmri external-model API.

Source
------
MDD_preproc/mdd_classifier/models/transformer.py            (TransformerEncoderModel)
MDD_preproc/mdd_classifier/models/multimodal_transformer.py (TimeSeriesTransformerEncoder)

`TransformerEncoderModel` is the generic single-modality encoder: one token per
time step, linear projection to d_model, standard nn.TransformerEncoder stack,
mean or CLS pooling, optional MLP head.

`TimeSeriesTransformerEncoder` is the ICA-time-course branch of
`MultiModalTransformerEncoder`, the model actually run by
`experiments_clf/modality_comparison.sh` (combination "T" and every combo
containing T). It is architecturally the same thing with a CLS token, dropout
after the input projection, and no MLP head — i.e. this class with
`pool="cls"`, `input_dropout=True`, `fc_hidden_dims=()`. That run used
d_model=384, nhead=8, num_layers=4, dim_feedforward=768, dropout=0.1, lr=3e-4
on 105 Neuromark components; see `MDDTransformer.from_modality_comparison`.

Adaptations for ml4fmri
-----------------------
* `input_dim`/`output_dim` renamed to the required `input_size`/`output_size`.
* `output_size` is the number of classes: the head emits `output_size` logits
  and training uses cross-entropy, instead of the single-logit BCE head used in
  MDD_preproc.
* `forward` returns `(logits, loss_load_dict)`; the pooled embedding stays
  available under "embedding".
* Added `lr`, `compute_loss`, `handle_batch`, `get_optimizer`,
  `prepare_dataloader` and `train_model`.

Note on sequence length
-----------------------
No positional encoding is used — this is faithful to both MDD_preproc classes.
With mean/CLS pooling and no positional information the encoder is permutation
invariant over time, so it reads the time course as a *set* of TRs. Worth
keeping in mind when comparing against the recurrent models.

Usage
-----
    from ml4fmri import cvbench
    from MDD_Transformer import MDDTransformer

    report = cvbench(DATA, LABELS,
                     models=["meanMLP"],
                     custom_models=MDDTransformer,
                     n_folds=5)

Rename the class to `Transformer` if you want it to override ml4fmri's built-in
Transformer rather than run alongside it.
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


class MDDTransformer(nn.Module):
    """
    TIME SERIES MODEL

    Transformer encoder with optional MLP head, from MDD_preproc.
    One token per time step; pooled by mean or by a prepended CLS token.
    Expected input shape: [batch_size, time_length, input_feature_size].
    Output: [batch_size, n_classes]
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: str = "relu",
        pool: Literal["mean", "cls"] = "mean",
        input_dropout: bool = False,
        fc_hidden_dims: Sequence[int] = (),
        lr: float = 3e-4,
    ):
        """
        Initialize the Transformer encoder.
        Args:
            input_size (int): Size of the vector at each time step in the input time series. \
                Common to all models, for FNC models it is the number of nodes in the FNC matrix.
            output_size (int): Number of classes for classification. Common to all models.
            d_model (int, hyperparameter): Transformer model dimension. Defaults to 128.
            nhead (int, hyperparameter): Number of attention heads. Defaults to 4.
            num_layers (int, hyperparameter): Number of encoder layers. Defaults to 2.
            dim_feedforward (int, hyperparameter): Feedforward dimension. Defaults to 256.
            dropout (float, hyperparameter): Dropout rate. Defaults to 0.1.
            activation (str, hyperparameter): Encoder activation, "relu" or "gelu". Defaults to "relu".
            pool (str, hyperparameter): "mean" over tokens, or "cls" for a prepended CLS token. \
                A CLS token is created whenever pool="cls". Defaults to "mean".
            input_dropout (bool, hyperparameter): Apply dropout right after the input projection, \
                as TimeSeriesTransformerEncoder does. Defaults to False.
            fc_hidden_dims (Sequence[int], hyperparameter): Hidden dims of the FC head. Defaults to ().
            lr (float, hyperparameter): Reference learning rate. Defaults to 3e-4 \
                (the LR used by modality_comparison.sh).
        """
        super().__init__()
        self.lr = lr

        if pool not in ("mean", "cls"):
            raise ValueError("pool must be 'mean' or 'cls'")
        self.pool = pool
        self.use_cls_token = pool == "cls"

        self.input_proj = nn.Linear(input_size, d_model)
        self.input_drop = nn.Dropout(dropout) if input_dropout else nn.Identity()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model)) if self.use_cls_token else None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        head_layers = []
        prev_dim = d_model
        for hidden_dim in fc_hidden_dims:
            head_layers.append(nn.Linear(prev_dim, hidden_dim))
            head_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        self.head = nn.Sequential(*head_layers)
        self.out = nn.Linear(prev_dim, output_size)

    @classmethod
    def from_modality_comparison(cls, input_size: int, output_size: int, **overrides):
        """
        The exact TimeSeriesTransformerEncoder configuration used by
        `experiments_clf/modality_comparison.sh` (d_model=384, nhead=8,
        num_layers=4, dim_feedforward=768, dropout=0.1, CLS pooling, no MLP head).

        Args:
            input_size (int): Feature size per time step (105 for Neuromark-105).
            output_size (int): Number of classes.
            **overrides: Any constructor argument to override.
        Returns:
            MDDTransformer
        """
        kwargs = dict(
            d_model=384,
            nhead=8,
            num_layers=4,
            dim_feedforward=768,
            dropout=0.1,
            pool="cls",
            input_dropout=True,
            fc_hidden_dims=(),
            lr=3e-4,
        )
        kwargs.update(overrides)
        return cls(input_size=input_size, output_size=output_size, **kwargs)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): (batch, seq_len, input_size). A 2-D input is treated \
                as a single time step.
        Returns:
            (logits, loss_load): logits of shape (batch, n_classes) and a dict \
                with "logits" and "embedding".
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.input_drop(self.input_proj(x))

        if self.use_cls_token:
            cls = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat([cls, x], dim=1)

        encoded = self.encoder(x)
        pooled = encoded[:, 0] if self.pool == "cls" else encoded.mean(dim=1)

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


if __name__ == "__main__":
    B, T, D, C = 8, 120, 53, 2
    x = torch.randn(B, T, D)
    models = {
        "MDDTransformer (mean)": MDDTransformer(input_size=D, output_size=C),
        "MDDTransformer (cls)": MDDTransformer(input_size=D, output_size=C, pool="cls"),
        "modality_comparison cfg": MDDTransformer.from_modality_comparison(D, C),
    }
    for name, model in models.items():
        logits, load = model(x)
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{name}: logits {tuple(logits.shape)}, "
              f"embedding {tuple(load['embedding'].shape)}, params {n}")
