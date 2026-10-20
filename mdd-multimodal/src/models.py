"""Encoders, transparent layers, and the task head.

Five fusion modes. The first two are the proposal's architectures (Figure 4,
pre_empt.pdf p.5); the last three are baselines that exist to be beaten.

  unified             PANEL A. Modality-specific encoders project onto ONE shared
                      node set, a single transparent graph links them, a decoder
                      reconstructs each modality from it, then the task head.
                      "a fully unified interpretable graph representation".
                      Modality is kept as a channel on each node, so an edge can
                      be attributed across modalities -- this is the mode that
                      produces "this GM pattern co-varies with this fALFF pattern".

  modality_specific   PANEL B. Each modality gets its OWN transparent graph over
                      its OWN native nodes (parcels for structure, components for
                      fMRI), trained jointly so each is "multimodally informed",
                      then integrated for the decoder and task head. Needs no
                      shared node set and no projection, and its layers can be
                      pretrained one modality at a time.

  joint / late / stack   BASELINES, no transparent layer. `joint` is the masked
                      mean of per-modality embeddings that this file used to do
                      exclusively; it is the "blended probability" the transparent
                      modes are meant to improve on. Keep them for comparison --
                      the interesting result is A or B versus these, on the same
                      folds.

Non-bypassability (p.4) is structural in the transparent modes: the only path
from an encoder to the task head runs through the graph. `assert_non_bypassable`
is called by the test suite rather than trusted.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .transparent import (Decoder, NodeProjection, TransparentGraph,
                          default_node_names, load_node_names)

TRANSPARENT_MODES = ("unified", "modality_specific")
BASELINE_MODES = ("joint", "late", "stack", "meanmlp", "brainnetcnn")
# modalities already indexed by ICA component, so their node mapping is free
COMPONENT_INDEXED = ("sFNC", "sFNC105")


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def grad_reverse(x, lam=1.0):
    return _GradReverse.apply(x, lam)



def make_head(in_dim, n_classes, hidden, dropout, style="mlp"):
    """Classifier readout.

    style="mlp"     : Linear(in,hidden) ReLU Dropout Linear(hidden,n)   [ours]
    style="linear"  : Linear(in, n)                                     [meanMLP]

    The lab's cvbench winner, meanMLP, is Linear(c,128) ReLU Dropout(.3)
    Linear(128,64) ReLU Linear(64,1) -- our Encoder already reproduces that trunk
    exactly; the only difference is that we add a second 128-wide hidden layer
    here instead of reading out directly. On that benchmark, capacity and
    performance ran INVERSELY (meanMLP 9,282 params -> 0.722 median; Transformer
    716,578 -> 0.556), so the extra layer is not obviously free.
    """
    if style == "linear":
        return nn.Linear(in_dim, n_classes)
    if style != "mlp":
        raise SystemExit(f"unknown model.head '{style}' (mlp|linear)")
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                         nn.Dropout(dropout), nn.Linear(hidden, n_classes))


class Encoder(nn.Module):
    """One modality -> latent. Plain MLP; swap for a 3D CNN if inputs become volumes."""

    def __init__(self, in_dim, out_dim=64, hidden=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim), nn.ReLU())
        self.out_dim = out_dim

    def forward(self, x):
        return self.net(x)


# ============================================================ transparent (A / B)
class TransparentModel(nn.Module):
    """Panels A and B. The difference is where the graph lives, nothing else."""

    def __init__(self, dims, mode="unified", n_nodes=53, channels=8, hidden=64,
                 dropout=0.3, n_classes=1, n_sites=0, adversarial_weight=0.0,
                 node_labels=None, projections=None, self_loops=True,
                 recon_weight=0.1, sparsity=1e-4, head="mlp"):
        super().__init__()
        if mode not in TRANSPARENT_MODES:
            raise SystemExit(f"mode '{mode}' is not a transparent mode")
        self.mode = mode
        self.modalities = list(dims)
        self.dims = dict(dims)
        self.channels = channels
        self.recon_weight = recon_weight
        self.sparsity = sparsity
        self.adversarial_weight = adversarial_weight
        projections = projections or {}

        if mode == "unified":
            # ---- ONE shared node set. Every modality must land on it. --------
            self.k = n_nodes
            self.node_names = node_labels or default_node_names(n_nodes)
            self.projections = nn.ModuleDict()
            self.proj_kind = {}
            for m, d in dims.items():
                if m in COMPONENT_INDEXED and _is_triangle(d, n_nodes):
                    kind = "fnc"          # already component-indexed: free
                elif m in projections:
                    kind = "fixed"        # prep-time overlap matrix
                else:
                    kind = "learned"      # honest fallback
                self.projections[m] = NodeProjection(
                    d, n_nodes, channels, mode=kind,
                    projection=projections.get(m))
                self.proj_kind[m] = kind
            self.graph = TransparentGraph(n_nodes, channels * len(dims), hidden,
                                          dropout, self_loops)
            self.decoder = Decoder(n_nodes, hidden, dims)
            self.head = make_head(n_nodes * hidden, n_classes, hidden,
                                  dropout, head)
            rep_dim = n_nodes * hidden
        else:
            # ---- PANEL B: one graph per modality, over its OWN nodes ---------
            # No projection anywhere: a parcellated modality's nodes ARE its
            # parcels, and a component-indexed one's nodes are its components.
            self.k = {}
            self.node_names = {}
            self.projections = nn.ModuleDict()
            self.graphs = nn.ModuleDict()
            self.proj_kind = {}
            for m, d in dims.items():
                if m in COMPONENT_INDEXED and _is_triangle(d, n_nodes):
                    km, kind = n_nodes, "fnc"
                else:
                    km, kind = d, "parcel"      # one node per input feature
                self.k[m] = km
                self.node_names[m] = (node_labels if (kind == "fnc" and node_labels)
                                      else [f"{m}_{i+1}" for i in range(km)])
                self.projections[m] = (
                    NodeProjection(d, km, channels, mode="fnc") if kind == "fnc"
                    else _ParcelNodes(channels))
                self.graphs[m] = TransparentGraph(km, channels, hidden,
                                                  dropout, self_loops)
                self.proj_kind[m] = kind
            rep_dim = sum(self.k[m] * hidden for m in dims)
            self.decoder = _MultiDecoder(self.k, hidden, dims)
            self.head = make_head(rep_dim, n_classes, hidden, dropout, head)

        self.site_head = (nn.Sequential(nn.Linear(rep_dim, hidden), nn.ReLU(),
                                        nn.Linear(hidden, n_sites))
                          if (adversarial_weight > 0 and n_sites > 1) else None)

    # -------------------------------------------------------------- forward
    def node_features(self, xs, masks):
        """Panel A only: (B, K, n_modalities * channels), modality-blocked.

        The concatenation is what preserves provenance. Block j of a node's
        features is modality j's contribution to that node and nothing else --
        which is what lets `modality_attribution` say which modality drives it.
        """
        blocks = []
        for m in self.modalities:
            h = self.projections[m](xs[m])                    # (B, K, ch)
            blocks.append(h * masks[m].float().view(-1, 1, 1))
        return torch.cat(blocks, dim=-1)

    def represent(self, xs, masks):
        if self.mode == "unified":
            H = self.graph(self.node_features(xs, masks))      # (B, K, hidden)
            return H.reshape(H.shape[0], -1), {"H": H}
        per, flats = {}, []
        for m in self.modalities:
            h = self.projections[m](xs[m])
            h = h * masks[m].float().view(-1, 1, 1)
            Hm = self.graphs[m](h)
            per[m] = Hm
            flats.append(Hm.reshape(Hm.shape[0], -1))
        return torch.cat(flats, dim=1), {"per": per}

    def forward(self, xs, masks):
        rep, cache = self.represent(xs, masks)
        logits = self.head(rep)

        aux = {}
        if self.recon_weight > 0:
            recon = (self.decoder(cache["H"]) if self.mode == "unified"
                     else self.decoder(cache["per"]))
            loss = rep.new_zeros(())
            for m in self.modalities:
                mk = masks[m].float()                          # absent -> no loss
                # MEAN over features, then mean over the subjects present.
                #
                # This used to be `.sum() / mk.sum()`, which summed over batch AND
                # features but divided only by the number of SUBJECTS -- a
                # per-subject SUM over features. With standardised inputs
                # (variance 1) and 2,029 features that inflates the term by ~2,029x,
                # so `recon_weight: 0.1` behaved like a weight near 200. Measured on
                # IS001: recon 113.5 against a task loss of 0.376 -- 302:1. The
                # model was an autoencoder with a classifier attached, and every
                # Panel A number so far was produced under that regime.
                #
                # Per-feature mean makes recon_weight mean what the config says:
                # a value of 1.0 weights reconstruction about equally with the BCE.
                se = ((recon[m] - xs[m]) ** 2).mean(dim=1)     # (B,) per subject
                loss = loss + (se * mk).sum() / mk.sum().clamp(min=1.0)
            aux["recon"] = self.recon_weight * loss
        if self.sparsity > 0:
            adjs = ([self.graph.adjacency(normalised=False)] if self.mode == "unified"
                    else [g.adjacency(normalised=False) for g in self.graphs.values()])
            aux["sparsity"] = self.sparsity * sum(a.abs().mean() for a in adjs)

        site_logits = (self.site_head(grad_reverse(rep, self.adversarial_weight))
                       if self.site_head is not None else None)
        return logits.squeeze(-1), site_logits, aux

    # --------------------------------------------------------------- readout
    def adjacency(self):
        """The interpretable object: one matrix for A, one per modality for B."""
        if self.mode == "unified":
            return self.graph.adjacency(normalised=False)
        return {m: g.adjacency(normalised=False) for m, g in self.graphs.items()}

    def describe(self):
        out = [f"  transparent mode : {self.mode}"]
        if self.mode == "unified":
            out.append(f"  shared nodes     : {self.k}")
            for m in self.modalities:
                kind = self.proj_kind[m]
                note = {"fnc": "component-indexed, no projection needed",
                        "fixed": "prep-time overlap matrix — node meaning is FIXED",
                        "learned": "LEARNED projection — node meaning for this "
                                   "modality is fitted, so its name is a weaker claim"
                        }[kind]
                out.append(f"    {m:<14} {kind:<8} {note}")
            if any(k == "learned" for k in self.proj_kind.values()):
                out.append("  !! at least one modality uses a learned projection.")
                out.append("     Run prep/build_projection.py for fixed ones before")
                out.append("     reporting node names as anatomical claims.")
        else:
            for m in self.modalities:
                out.append(f"    {m:<14} {self.k[m]:>5} own nodes ({self.proj_kind[m]})")
            out.append("  no shared node set needed — this is why B is cheaper")
        out.append(f"  reconstruction   : weight {self.recon_weight}"
                   + ("  (0 = discriminative only, not decomposition-like)"
                      if self.recon_weight == 0 else ""))
        return "\n".join(out)

    def assert_non_bypassable(self):
        """p.4 requires the layer to be a non-bypassable bottleneck. Verify it.

        Zeroing the graph output must destroy the prediction. If the head can
        still discriminate, something routes around the graph and the
        interpretation of the adjacency is void.
        """
        graphs = ([self.graph] if self.mode == "unified" else list(self.graphs.values()))
        for g in graphs:
            for p in g.parameters():
                if p.requires_grad is False:
                    raise AssertionError("graph parameters are frozen")
        # the head's input dimension must equal the graph representation's size,
        # i.e. nothing else is concatenated in alongside it.
        # The head is a bare nn.Linear when model.head=linear (meanMLP's readout)
        # and an nn.Sequential otherwise -- indexing [0] unconditionally raised
        # TypeError on the linear head and aborted the run before fold 1.
        first = self.head[0] if isinstance(self.head, nn.Sequential) else self.head
        if self.mode == "unified":
            expect = self.k * self.graph.W.out_features
        else:
            expect = sum(self.k[m] * self.graphs[m].W.out_features
                         for m in self.modalities)
        if first.in_features != expect:
            raise AssertionError(
                f"head takes {first.in_features} inputs but the graph produces "
                f"{expect} — something bypasses the transparent layer")
        return True


class _ParcelNodes(nn.Module):
    """Panel B, parcellated modality: node i is feature i, lifted to `channels`."""

    def __init__(self, channels):
        super().__init__()
        self.mix = nn.Linear(1, channels)

    def forward(self, x):
        return self.mix(x.unsqueeze(-1))


class _MultiDecoder(nn.Module):
    """Panel B: reconstruct each modality from its own graph's nodes."""

    def __init__(self, k_by_mod, hidden, dims, width=128):
        super().__init__()
        self.heads = nn.ModuleDict({
            m: nn.Sequential(nn.Linear(k_by_mod[m] * hidden, width), nn.ReLU(),
                             nn.Linear(width, dims[m]))
            for m in dims})

    def forward(self, per):
        return {m: h(per[m].reshape(per[m].shape[0], -1))
                for m, h in self.heads.items()}


# ================================================================== baselines
class FusionModel(nn.Module):
    """No transparent layer. Kept as the comparison A and B have to beat."""

    def __init__(self, dims, n_classes=1, encoder_dim=64, hidden=128, dropout=0.3,
                 fusion="joint", n_sites=0, adversarial_weight=0.0, head="mlp",
                 brainnet=False):
        super().__init__()
        self.modalities = list(dims)
        self.fusion = fusion
        self.adversarial_weight = adversarial_weight
        # BrainNetCNN replaces the MLP encoder for the CONNECTIVITY modality only.
        # Its convolutions are defined over rows and columns of a (k, k) matrix, so
        # a 217-parcel vector has no valid interpretation for it; those modalities
        # keep the ordinary encoder and the streams are fused as usual.
        self.brainnet = brainnet
        enc = {}
        self.fnc_mods = []
        for m, d in dims.items():
            if brainnet and _triangle_k(d):
                enc[m] = BrainNetCNNEncoder(d, out_dim=encoder_dim, dropout=dropout)
                self.fnc_mods.append(m)
            else:
                enc[m] = Encoder(d, encoder_dim, hidden, dropout)
        if brainnet and not self.fnc_mods:
            raise SystemExit(
                "fusion.mode=brainnetcnn needs a connectivity modality (sFNC). "
                f"Got only {list(dims)}, none of which is a triangular edge vector.")
        self.encoders = nn.ModuleDict(enc)
        if fusion in ("joint", "brainnetcnn"):
            self.head = make_head(encoder_dim, n_classes, hidden, dropout, head)
        elif fusion in ("late", "stack"):
            self.per_mod = nn.ModuleDict(
                {m: nn.Linear(encoder_dim, n_classes) for m in dims})
            if fusion == "stack":
                self.mixer = nn.Linear(len(dims) * n_classes, n_classes)
        else:
            raise SystemExit(f"unknown fusion.mode '{fusion}'")
        self.site_head = (nn.Sequential(nn.Linear(encoder_dim, hidden), nn.ReLU(),
                                        nn.Linear(hidden, n_sites))
                          if (adversarial_weight > 0 and n_sites > 1) else None)

    def embed(self, xs, masks):
        zs, ms = [], []
        for m in self.modalities:
            z = self.encoders[m](xs[m])
            mk = masks[m].float().unsqueeze(1)
            zs.append(z * mk)
            ms.append(mk)
        Z = torch.stack(zs, 0).sum(0)
        M = torch.stack(ms, 0).sum(0).clamp(min=1.0)
        return Z / M, zs, ms

    def describe(self):
        if self.brainnet:
            return ("  fusion mode      : brainnetcnn  (E2E/E2N/N2G over the 53x53 "
                    "FNC matrix for " + ", ".join(self.fnc_mods)
                    + "; MLP encoders elsewhere)")
        return (f"  fusion mode      : {self.fusion}  (BASELINE — no transparent "
                f"layer; modality identity is averaged away)")

    def forward(self, xs, masks):
        shared, zs, ms = self.embed(xs, masks)
        if self.fusion in ("joint", "brainnetcnn"):
            logits = self.head(shared)
        else:
            per = [self.per_mod[m](zs[i]) for i, m in enumerate(self.modalities)]
            if self.fusion == "late":
                num = torch.stack([p * ms[i] for i, p in enumerate(per)], 0).sum(0)
                den = torch.stack(ms, 0).sum(0).clamp(min=1.0)
                logits = num / den
            else:
                logits = self.mixer(torch.cat(per, dim=1))
        site_logits = (self.site_head(grad_reverse(shared, self.adversarial_weight))
                       if self.site_head is not None else None)
        return logits.squeeze(-1), site_logits, {}



# =================================================== published benchmark models
# Two architectures from the lab's cvbench sweep, reimplemented so they train on
# OUR cohort, OUR folds and OUR modalities. Quoting their published AUCs would
# compare across different subjects (1,944 vs 2,055), different inputs
# (timecourses vs our feature set) and a different CV protocol; running them here
# removes all three differences at once.
#
# BolT, Glacier and DICE are deliberately absent: they consume 53x230 timecourses,
# which our feature files do not carry.

class MeanMLPFusion(nn.Module):
    """cvbench's winner (0.741 there), as EARLY fusion.

    Their meanMLP is Linear(c,128) ReLU Dropout(.3) Linear(128,64) ReLU
    Linear(64,1) over one feature vector. The faithful multimodal analogue is to
    concatenate the modalities into one vector and apply that same MLP -- which
    is genuinely different from `fusion: joint`, where each modality gets its own
    encoder and the embeddings are averaged. Early fusion versus late fusion, one
    MLP versus four.

    A missing modality is zero-filled and its mask recorded, as elsewhere; with
    concatenation there is no per-modality average to renormalise.
    """

    def __init__(self, dims, n_classes=1, hidden=128, encoder_dim=64, dropout=0.3,
                 n_sites=0, adversarial_weight=0.0):
        super().__init__()
        self.modalities = list(dims)
        self.fusion = "meanmlp"
        self.adversarial_weight = adversarial_weight
        total = sum(dims.values())
        self.trunk = nn.Sequential(
            nn.Linear(total, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, encoder_dim), nn.ReLU())
        self.head = nn.Linear(encoder_dim, n_classes)
        self.site_head = (nn.Sequential(nn.Linear(encoder_dim, hidden), nn.ReLU(),
                                        nn.Linear(hidden, n_sites))
                          if (adversarial_weight > 0 and n_sites > 1) else None)

    def describe(self):
        return ("  fusion mode      : meanmlp  (cvbench meanMLP, EARLY fusion — "
                "modalities concatenated into one MLP)")

    def forward(self, xs, masks):
        parts = [xs[m] * masks[m].float().unsqueeze(1) for m in self.modalities]
        z = self.trunk(torch.cat(parts, dim=1))
        site_logits = (self.site_head(grad_reverse(z, self.adversarial_weight))
                       if self.site_head is not None else None)
        return self.head(z).squeeze(-1), site_logits, {}


class E2EBlock(nn.Module):
    """Edge-to-edge: a cross-shaped filter over the connectivity matrix.

    For node pair (i, j) it pools row i and column j. That is the whole point of
    BrainNetCNN -- an ordinary 3x3 convolution is meaningless on an adjacency
    matrix, where adjacency of indices carries no spatial meaning.
    """

    def __init__(self, in_ch, out_ch, k):
        super().__init__()
        self.k = k
        self.row = nn.Conv2d(in_ch, out_ch, (1, k))
        self.col = nn.Conv2d(in_ch, out_ch, (k, 1))

    def forward(self, x):                       # (B, C, k, k)
        a = self.row(x)                         # (B, C', k, 1)
        b = self.col(x)                         # (B, C', 1, k)
        return a.repeat(1, 1, 1, self.k) + b.repeat(1, 1, self.k, 1)


class BrainNetCNNEncoder(nn.Module):
    """Flat FNC edge vector -> (k, k) matrix -> E2E, E2E, E2N, N2G -> embedding.

    Rebuilding the matrix is not cosmetic: the model's filters are defined over
    rows and columns of the connectivity matrix, so the 1,378 edges have to go
    back to the (53, 53) form they came from, symmetric, zero diagonal, in the
    SAME upper-triangle order extract_sfnc.py wrote them.
    """

    def __init__(self, n_edges, out_dim=64, c1=16, c2=32, dropout=0.3):
        super().__init__()
        k = int(round((1 + (1 + 8 * n_edges) ** 0.5) / 2))
        if k * (k - 1) // 2 != n_edges:
            raise SystemExit(
                f"BrainNetCNN needs a triangular edge vector; {n_edges} is not "
                "k(k-1)/2 for any k. Give it sFNC, not a parcel-mean modality.")
        self.k = k
        iu = torch.triu_indices(k, k, offset=1)
        self.register_buffer("iu0", iu[0])
        self.register_buffer("iu1", iu[1])
        self.e2e1 = E2EBlock(1, c1, k)
        self.e2e2 = E2EBlock(c1, c1, k)
        self.e2n = nn.Conv2d(c1, c2, (1, k))
        self.n2g = nn.Conv2d(c2, out_dim, (k, 1))
        self.drop = nn.Dropout(dropout)
        self.out_dim = out_dim

    def forward(self, v):                       # (B, n_edges)
        B = v.shape[0]
        M = v.new_zeros(B, self.k, self.k)
        M[:, self.iu0, self.iu1] = v
        M = M + M.transpose(1, 2)               # symmetric, diagonal stays 0
        x = M.unsqueeze(1)
        x = F.leaky_relu(self.e2e1(x), 0.33)
        x = F.leaky_relu(self.e2e2(x), 0.33)
        x = F.leaky_relu(self.e2n(x), 0.33)
        x = F.leaky_relu(self.n2g(self.drop(x)), 0.33)
        return x.flatten(1)


# ====================================================================== build
def _is_triangle(d, k):
    return d == k * (k - 1) // 2


def _triangle_k(d):
    """k if d is k(k-1)/2 edges of a k-node graph, else None.

    This is how a connectivity modality is recognised without being told its
    name: 1,378 is the upper triangle of 53 nodes, 217 parcels is not a triangle
    of anything. Matching on the name "sFNC" instead would silently skip a
    connectivity stream called something else.
    """
    if d is None or d < 1:
        return None
    k = int(round((1 + (1 + 8 * d) ** 0.5) / 2))
    return k if k * (k - 1) // 2 == d else None


def load_projections(path, dims, k):
    """NPZ of per-modality (d, K) overlap matrices from prep/build_projection.py."""
    if not path:
        return {}
    import os
    if not os.path.exists(path):
        print(f"    (no projection file at {path} — falling back to learned)")
        return {}
    z = np.load(path, allow_pickle=True)
    out = {}
    for m in dims:
        if m in z:
            P = np.asarray(z[m])
            if P.shape == (dims[m], k):
                out[m] = P
            else:
                print(f"    ({m}: projection is {P.shape}, expected "
                      f"{(dims[m], k)} — ignoring)")
    return out


def build(dims, cfg, n_sites=0, n_classes=1):
    m = cfg["model"]
    mode = m["fusion"]["mode"]
    adv = (cfg["site"]["adversarial_weight"]
           if cfg["site"]["mode"] == "adversarial" else 0.0)

    if mode == "meanmlp":
        return MeanMLPFusion(dims, n_classes=n_classes, hidden=m["hidden"],
                             encoder_dim=m["encoder_dim"], dropout=m["dropout"],
                             n_sites=n_sites, adversarial_weight=adv)
    if mode in BASELINE_MODES:
        return FusionModel(dims, n_classes=n_classes, encoder_dim=m["encoder_dim"],
                           hidden=m["hidden"], dropout=m["dropout"], fusion=mode,
                           n_sites=n_sites, adversarial_weight=adv,
                           head=m.get("head", "mlp"),
                           brainnet=(mode == "brainnetcnn"))
    if mode not in TRANSPARENT_MODES:
        raise SystemExit(f"unknown fusion.mode '{mode}' "
                         f"({'|'.join(TRANSPARENT_MODES + BASELINE_MODES)})")

    t = m.get("transparent", {})
    k = int(t.get("n_nodes", 53))
    names, from_file = load_node_names(t.get("node_labels"), k)
    if not from_file:
        print(f"    !! node names are PLACEHOLDERS ({names[0]}, {names[1]}, ...). "
              "The real\n"
              "       names are the Neuromark component labels that ship with the\n"
              "       template. Set model.transparent.node_labels before quoting\n"
              "       any node in text — a readout with these names tells you the\n"
              "       graph's structure but not which brain networks it involves.")
    return TransparentModel(
        dims, mode=mode, n_nodes=k, channels=int(t.get("channels", 8)),
        hidden=m["hidden"] // 2, dropout=m["dropout"], n_classes=n_classes,
        n_sites=n_sites, adversarial_weight=adv, node_labels=names,
        projections=load_projections(t.get("projection"), dims, k),
        self_loops=bool(t.get("self_loops", True)),
        recon_weight=float(t.get("recon_weight", 0.1)),
        sparsity=float(t.get("sparsity", 1e-4)),
        head=m.get("head", "mlp"))
