"""Transparent layers — panel A of Figure 4 (pre_empt.pdf p.5).

What the proposal asks for
--------------------------
p.4: "interpretable differentiable layers -- specifically, utilizing structured
frameworks like graphs and sorted arrays. These layers are designed to serve as
critical, NON-BYPASSABLE information bottlenecks."

p.6: "nodes in the graph within the transparent layer correspond to clearly
interpretable brain regions or other clinically relevant attributes ... n spatial
components, e.g., ICA-based spatial components or regions of an atlas."

p.5, panel A: "integrates modality-specific encoders into a joint transparent
representation, which serves as the input to a decoder that subsequently feeds
into the task module ... a fully unified interpretable graph representation."

Three commitments follow, and they are what this module enforces.

1. NODES HAVE FIXED IDENTITY. K nodes, named, supplied from outside. Not learned,
   not permuted, not re-ordered between runs. Node 7 is the same brain thing in
   every model you train. Without this the graph is decorative.

2. THE LAYER IS NON-BYPASSABLE. There is exactly one path from the encoders to
   the task head and it goes through the node representation. No skip connection,
   no auxiliary route. `assert_non_bypassable()` checks this by construction and
   the test suite calls it.

3. MODALITY SURVIVES AS A CHANNEL ON EACH NODE. This is the design choice that
   makes panel A worth building rather than just deep. Each node carries
   `n_modalities x channels_per_modality` features, so modality provenance is not
   averaged away. An edge between node i and node j can then be attributed: if
   node i's active channels are GM and node j's are FALFF, the edge literally
   states "this grey-matter pattern co-varies with this fALFF pattern". That is
   the sentence the whole exercise is for.

   The alternative -- averaging modalities into one vector per node -- is what
   src/models.py did before, and it destroys exactly this information.

The shared node set problem
---------------------------
Panel A needs every modality on ONE node set. Ours are not: sFNC is indexed by
Neuromark components, the structural volumes by atlas parcels. Two ways across,
and the honest difference between them matters:

  FIXED projection (`projection` in the config). A (d_modality, K) matrix computed
  at prep time from the Neuromark spatial maps -- how much each parcel overlaps
  each component. The nodes then mean what the template says they mean, and the
  readout is trustworthy. This is the correct version. See the note in
  prep/build_projection.py.

  LEARNED projection (the default, because it runs today). A linear map the
  optimiser fits. The node set is still shared and still fixed in number, but what
  a node MEANS for a modality lacking a fixed projection is learned, so the name
  attached to it is a weaker claim. Reported honestly by `describe()` rather than
  quietly assumed.

sFNC is the one modality that needs no projection either way: it is already
indexed by component pairs, so component i's row of the FNC matrix is node i's
natural feature vector. `fnc_rows_to_nodes` does that exactly.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Neuromark 1.0 is 53 components in 7 domains, with this composition as published
# (SC 5, AUD 2, SM 9, VIS 9, CC 17, DM 7, CB 4). Recorded ONLY to check the length
# of a labels file you supply. It is deliberately NOT used to generate names,
# because doing so requires assuming the template's component ORDER matches this
# domain order -- and if that assumption is wrong, node 7 gets confidently
# labelled "SM_2" when it is actually a visual component. A wrong anatomical name
# is worse than an obviously absent one: it survives into a figure caption.
#
# The real names are the Neuromark component labels, which ship with the template
# (and are in the supplementary table of Du et al., NeuroImage: Clinical 2020).
# Point model.transparent.node_labels at them.
NM1_DOMAINS = [("SC", 5), ("AUD", 2), ("SM", 9), ("VIS", 9),
               ("CC", 17), ("DM", 7), ("CB", 4)]
NM1_TOTAL = sum(n for _, n in NM1_DOMAINS)


def default_node_names(k):
    """Unmistakably-placeholder names. Never guesses anatomy."""
    return [f"node_{i+1:02d}" for i in range(k)]


def parse_neuromark_domains(path):
    """Read Neuromark's own domain file, e.g. Network_templates/NeuroMark1/
    Neuromark_fMRI_1.0.txt

        SC, 1, 2, 3, 4, 5
        AU, 6, 7
        SM, 8, 9, 10, 11, 12, 13, 14, 15, 16
        VI, 17, 18, 19, 20, 21, 22, 23, 24, 25
        CC, 26, ... , 42
        DM, 43, ... , 49
        CB, 50, 51, 52, 53

    Returns names indexed by component, e.g. component 7 -> "AU_07". The
    component NUMBER is kept in the name rather than a within-domain counter, so
    a node in a figure can be traced back to the template without a lookup.

    This file gives the DOMAIN of each component, not an anatomical name. So
    "CC_31" is a verified claim ("component 31, in the cognitive-control domain")
    while "right dorsolateral prefrontal cortex" is not available from here --
    that would need the per-component peak labels.
    """
    domains = {}
    order = []
    with open(path) as f:
        for line in f:
            parts = [t.strip() for t in line.replace("\t", ",").split(",") if t.strip()]
            if len(parts) < 2:
                continue
            dom = parts[0]
            for tok in parts[1:]:
                try:
                    idx = int(tok)
                except ValueError:
                    continue
                domains[idx] = dom
                order.append(idx)
    if not domains:
        return None
    lo, hi = min(domains), max(domains)
    missing = [i for i in range(lo, hi + 1) if i not in domains]
    if missing:
        raise SystemExit(
            f"{path}: components {missing} have no domain. The file looks "
            f"truncated — it should cover {lo}..{hi} with no gaps.")
    if len(order) != len(set(order)):
        raise SystemExit(f"{path}: a component appears in more than one domain")
    names = [f"{domains[i]}_{i:02d}" for i in range(lo, hi + 1)]

    # Neuromark ships a sibling "_indexall.txt" giving each component's index in
    # the ORIGINAL decomposition (e.g. "SC, 69 53 98 99 45" -> template component
    # 1 is original component 69). It is not needed to name nodes, but the two
    # files must agree on the domain composition; if they disagree, one of them
    # is not the file we think it is, and every node label is suspect.
    sib = path.replace(".txt", "_indexall.txt")
    if os.path.exists(sib) and sib != path:
        orig = {}
        with open(sib) as f:
            for line in f:
                parts = line.replace(",", " ").split()
                if len(parts) < 2:
                    continue
                dom, idxs = parts[0], [int(t) for t in parts[1:] if t.lstrip("-").isdigit()]
                orig.setdefault(dom, []).extend(idxs)
        here = {}
        for i in range(lo, hi + 1):
            here.setdefault(domains[i], []).append(i)
        bad = [d for d in here if len(orig.get(d, [])) != len(here[d])]
        if bad:
            raise SystemExit(
                f"{os.path.basename(path)} and {os.path.basename(sib)} disagree on "
                f"domain size for {bad}. One of these is not the file it appears "
                "to be — do not label nodes until this is resolved.")
        flat = [ix for d in here for ix in orig[d]]
        print(f"    cross-checked against {os.path.basename(sib)}: "
              f"{len(flat)} original ICA indices, composition agrees")
        print(f"       traceability: template 1 = original {flat[0]}, "
              f"template {hi} = original {flat[-1]}")
    return names


def load_node_names(path, k):
    """Neuromark domain file, one name per line, or a CSV with a 'name' column."""
    if not path:
        return default_node_names(k), False
    if not os.path.exists(path):
        print(f"    (no node labels at {path} — using placeholder names)")
        return default_node_names(k), False
    # Neuromark's own format first: "DOMAIN, idx, idx, ..." per line
    try:
        nm = parse_neuromark_domains(path)
    except SystemExit:
        raise                       # a real disagreement: surface it
    except (ValueError, UnicodeDecodeError, OSError):
        nm = None                   # not this format; try the others below
    # NOTE: deliberately NOT a bare `except Exception`. A missing import inside
    # the parser once got swallowed here and the loader quietly fell back to
    # reading the file as one-name-per-line, producing 7 names for a 53-node
    # layer. A bug in the parser must crash, not degrade.
    if nm:
        if len(nm) != k:
            raise SystemExit(
                f"{path} describes {len(nm)} components but the layer has {k} "
                f"nodes. Set model.transparent.n_nodes={len(nm)} to match the "
                "template, or point at the template for the right model order.")
        print(f"    node labels: {len(nm)} components from {os.path.basename(path)} "
              f"({nm[0]}, {nm[1]}, ..., {nm[-1]})")
        print("       these are DOMAIN labels — verified from the template — not "
              "anatomical names")
        return nm, True

    if path.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(path)
        col = next((c for c in df.columns if c.lower() in ("name", "label",
                                                           "component", "node")), None)
        if col is None:
            raise SystemExit(f"{path} has no name/label/component column")
        names = df[col].astype(str).tolist()
        if "domain" in {c.lower() for c in df.columns}:
            dcol = next(c for c in df.columns if c.lower() == "domain")
            names = [f"{d}_{n}" for d, n in zip(df[dcol].astype(str), names)]
    else:
        with open(path) as f:
            names = [ln.strip() for ln in f if ln.strip()]
    if len(names) != k:
        raise SystemExit(f"{path} has {len(names)} names but the layer has {k} nodes")
    if k == NM1_TOTAL:
        print(f"    node labels: {len(names)} loaded from {path} "
              f"({names[0]}, {names[1]}, ..., {names[-1]})")
    return names, True


def fnc_rows_to_nodes(x, k):
    """(B, k*(k-1)/2) upper-triangle FNC -> (B, k, k) where row i is node i.

    sFNC needs no projection: it is already indexed by component pairs, so
    component i's connectivity profile IS node i's feature vector. The diagonal
    is left at zero (self-connectivity is not in the vectorised triangle).
    """
    b = x.shape[0]
    iu = torch.triu_indices(k, k, offset=1, device=x.device)
    A = torch.zeros(b, k, k, dtype=x.dtype, device=x.device)
    A[:, iu[0], iu[1]] = x
    return A + A.transpose(1, 2)


class NodeProjection(nn.Module):
    """One modality's features -> (B, K, channels) on the shared node set.

    mode='fnc'      the modality IS component-indexed; rows become node features
    mode='fixed'    a prep-time (d, K) overlap matrix; nodes mean what it says
    mode='learned'  a fitted linear map; the node set is shared, the meaning is not
    """

    def __init__(self, in_dim, k, channels=8, mode="learned", projection=None):
        super().__init__()
        self.k, self.channels, self.mode = k, channels, mode
        if mode == "fnc":
            # k features per node (its FNC row) -> channels
            self.mix = nn.Linear(k, channels)
        elif mode == "fixed":
            if projection is None:
                raise SystemExit("mode='fixed' needs a projection matrix")
            P = torch.as_tensor(np.asarray(projection), dtype=torch.float32)
            if P.shape != (in_dim, k):
                raise SystemExit(f"projection is {tuple(P.shape)}, expected ({in_dim}, {k})")
            # rows sum to 1 so a node is a weighted average of its parcels
            P = P / P.sum(dim=0, keepdim=True).clamp(min=1e-8)
            self.register_buffer("P", P)        # buffer, not parameter: FIXED
            self.mix = nn.Linear(1, channels)
        elif mode == "learned":
            self.proj = nn.Linear(in_dim, k * channels)
        else:
            raise SystemExit(f"unknown projection mode '{mode}' (fnc|fixed|learned)")

    def forward(self, x):
        if self.mode == "fnc":
            return self.mix(fnc_rows_to_nodes(x, self.k))          # (B, K, ch)
        if self.mode == "fixed":
            nodal = x @ self.P                                      # (B, K)
            return self.mix(nodal.unsqueeze(-1))                    # (B, K, ch)
        return self.proj(x).view(-1, self.k, self.channels)


class TransparentGraph(nn.Module):
    """The bottleneck: a learnable graph over K named nodes.

    The adjacency is a single (K, K) parameter, symmetrised and given self-loops
    -- the self-loops are drawn in Figure 4 and they matter, because a node's own
    contribution should be readable separately from its edges.

    One graph-convolution step: H' = ReLU( norm(A) @ H @ W ). Deliberately one
    step and one weight matrix. Stacking more would let information route around
    any single edge and make the adjacency harder to read, which defeats the point.
    """

    def __init__(self, k, in_ch, hidden, dropout=0.3, self_loops=True):
        super().__init__()
        self.k = k
        self.self_loops = self_loops
        # init near zero so the graph starts almost uninformative and has to earn
        # its edges, rather than starting at a dense random structure
        self.adj_raw = nn.Parameter(torch.randn(k, k) * 0.01)
        self.W = nn.Linear(in_ch, hidden)
        self.drop = nn.Dropout(dropout)

    def adjacency(self, normalised=True):
        """The interpretable object. Symmetric, non-negative, with self-loops."""
        A = F.softplus(self.adj_raw)
        A = 0.5 * (A + A.t())                       # undirected
        if self.self_loops:
            A = A + torch.diag(torch.diagonal(A))   # emphasise the diagonal
        else:
            A = A - torch.diag(torch.diagonal(A))
        if not normalised:
            return A
        d = A.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return A / d.sqrt() / d.sqrt().t()

    def forward(self, H):
        return self.drop(F.relu(self.adjacency() @ self.W(H)))


class Decoder(nn.Module):
    """Shared node representation -> reconstruct each modality's input.

    Panel A is the only one of the four drawn with a decoder, and that is what
    makes it the analogue of a decomposition: if every modality can be rebuilt
    from the shared nodes, the nodes are carrying the modalities jointly rather
    than just enough of them to classify. The reconstruction loss is what applies
    that pressure; `recon_weight: 0` turns A into a purely discriminative model,
    which is a useful ablation but no longer decomposition-like.
    """

    def __init__(self, k, hidden, dims, width=128):
        super().__init__()
        self.heads = nn.ModuleDict({
            m: nn.Sequential(nn.Linear(k * hidden, width), nn.ReLU(),
                             nn.Linear(width, d))
            for m, d in dims.items()})

    def forward(self, H):
        flat = H.reshape(H.shape[0], -1)
        return {m: head(flat) for m, head in self.heads.items()}


# ------------------------------------------------------------------ readout
def top_edges(adj, node_names, n=20, exclude_self=True):
    """Strongest edges, named. The model's claim, in a form you can read."""
    A = adj.detach().cpu().numpy().copy()
    if exclude_self:
        np.fill_diagonal(A, 0.0)
    iu = np.triu_indices(A.shape[0], k=1)
    order = np.argsort(-np.abs(A[iu]))[:n]
    return [(node_names[iu[0][j]], node_names[iu[1][j]], float(A[iu[0][j], iu[1][j]]))
            for j in order]


def modality_attribution(model, xs, masks, node_names, top_k=10):
    """Which modality drives each node, by gradient of node activation wrt input.

    This is what turns an edge into the cross-modal sentence. For each node we
    ask how much each modality's channels contribute; an edge between a node
    dominated by GM and a node dominated by FALFF is the statement the proposal
    is after.

    Uses channel magnitude rather than gradients: the channel block for modality
    m at node i is exactly that modality's contribution to that node, because the
    channels are concatenated and never mixed before the graph layer.
    """
    model.eval()
    with torch.no_grad():
        H = model.node_features(xs, masks)                   # (B, K, M*ch)
    ch = model.channels
    out = {}
    for j, m in enumerate(model.modalities):
        block = H[:, :, j * ch:(j + 1) * ch]                 # (B, K, ch)
        out[m] = block.abs().mean(dim=(0, 2)).cpu().numpy()  # (K,)
    total = np.stack([out[m] for m in model.modalities], 0).sum(0) + 1e-12
    rows = []
    for i, name in enumerate(node_names):
        share = {m: float(out[m][i] / total[i]) for m in model.modalities}
        lead = max(share, key=share.get)
        rows.append((name, lead, share))
    return rows
