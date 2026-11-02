"""
DGCNN (Zhang et al. 2018, AAAI) on FNC graphs for ml4fmri's cvbench, as Zhu et al. 2023
(Front. Hum. Neurosci. 17:1094592) used it for MDD vs HC; sizes, batch and lr follow them.

Nodes are components, with their FNC rows as features. DGCNN keeps the edges with
FNC > 0.3, like the paper; DGCNNRaw uses the raw, signed FNC as edge weights.
"""

import torch
from torch import nn

from ml4fmri.models.helper_functions import (
    BasicTrainer,
    basic_Adam_optimizer,
    basic_ce_loss,
    basic_dataloader,
    basic_handle_batch,
)


class DGCNN(nn.Module):
    """FNC MODEL. Input [batch, n_nodes, n_nodes] FNC; output [batch, n_classes] logits."""

    def __init__(self, input_size, output_size, threshold=0.3, k=None,
                 gcn_sizes=(32, 32, 32, 1), lr=1e-3):
        super().__init__()
        self.lr, self.threshold = lr, threshold
        self.k = k or input_size  # Zhang et al.'s rule keeps every node when graphs share a size

        sizes = (input_size, *gcn_sizes)
        self.gcn = nn.ModuleList(nn.Linear(a, b, bias=False) for a, b in zip(sizes, sizes[1:]))
        width = sum(gcn_sizes)  # channels per node once the layers are concatenated

        # one conv step per sorted node, then pool and a second conv
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, width, stride=width), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5), nn.ReLU(), nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(32 * (self.k // 2 - 4), 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, output_size),
        )

    def adjacency(self, fnc):
        """D^-1 (A + I), as in Zhang et al.; A is FNC > threshold, or the raw FNC."""
        eye = torch.eye(fnc.shape[-1], device=fnc.device)
        a = fnc * (1 - eye)
        if self.threshold is not None:
            a = (a > self.threshold).float()
        a = a + eye
        return a / a.abs().sum(-1, keepdim=True)  # |weights|: raw FNC has negative edges

    def forward(self, fnc):
        adj, h = self.adjacency(fnc), fnc  # node features: each node's FNC row
        layers = []
        for linear in self.gcn:
            h = torch.tanh(adj @ linear(h))
            layers.append(h)
        h = torch.cat(layers, -1)

        # SortPooling: order nodes by the last channel, keep the top k
        order = h[..., -1].argsort(dim=1, descending=True)[:, :self.k]
        h = h.gather(1, order.unsqueeze(-1).expand(-1, -1, h.shape[-1]))

        logits = self.head(self.conv(h.flatten(1).unsqueeze(1)))
        return logits, {"logits": logits}

    #### ml4fmri training API ####

    def compute_loss(self, loss_load, targets):
        return basic_ce_loss(loss_load["logits"], targets)

    def handle_batch(self, batch):
        return basic_handle_batch(self, batch)

    def get_optimizer(self, lr=None):
        return basic_Adam_optimizer(self, lr or self.lr)

    @staticmethod
    def prepare_dataloader(data, labels, batch_size=50, shuffle=True):
        # Pearson FNC per subject from its time courses; batch 50 as in the paper
        return basic_dataloader(data, labels, type="FNC", batch_size=batch_size, shuffle=shuffle)

    def train_model(self, train_loader, val_loader, test_loader, epochs=200, lr=None,
                    device=None, patience=30):
        return BasicTrainer(
            model=self, train_loader=train_loader, val_loader=val_loader,
            test_loader=test_loader, epochs=epochs, lr=lr, device=device, patience=patience,
        ).run()


class DGCNNRaw(DGCNN):
    """DGCNN with the raw, signed FNC as edge weights instead of FNC > 0.3."""

    def __init__(self, input_size, output_size, **kwargs):
        super().__init__(input_size, output_size, threshold=None, **kwargs)
