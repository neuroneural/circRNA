# mdd-multimodal

Multimodal deep-learning fusion for DIRECT Phase 2 (MDD_DIRECT, n = 3,525).

Structure follows `multimodal-subnetworks` (Joanne): `conf/` + `src/` + `prep/` +
`scripts/`, config-driven, one entry point, SLURM wrappers.

---

## Methods — traceable to the PRE-EMPT proposal

Every modelling component below is named in `ProjectNarrative--Overall_100324.pdf`.
Nothing is included that the proposal does not name.

| proposal method | page | where it lives here |
|---|---|---|
| **transparent layers** | pre_empt p.4, 5, 6 | `src/transparent.py` — graph over named nodes, non-bypassable bottleneck; `src/readout.py` reads it |
| **deep learning** | 1, 4, 7 | `src/models.py` — encoders, panels A and B, baselines |
| **data fusion** / multimodal fusion | 7 | `src/models.py` — `FusionModel`, `fusion.mode` |
| **dynamic connectivity** | 4 | `prep/extract_dfnc.py`, modality `dfnc` |
| **joint information** | 4 | `fusion.mode: joint` — modalities share a latent space and are trained jointly, rather than combined after the fact |
| **group ICA (GIFT)** | 16 | sFNC / spectra features, back-reconstructed Neuromark components |
| **linear mixed effects models** | 16, 24, 32 | `src/site.py` — site as a **random effect**; `src/evaluate.py` — LME on fold results |
| **correlation analyses** | throughout | `src/evaluate.py` |
| **predictive analytics** | 1 | the whole pipeline; `src/evaluate.py` reports AUC + CI |

Deliberately **not** included: joint ICA / mCCA+jICA (not named in the proposal),
and any architecture the proposal does not specify (no transformer, no GNN).

---

## The architecture

Two of the proposal's four panels are implemented, switched by one config key:

```yaml
model:
  fusion:
    mode: unified            # panel A
    # mode: modality_specific  # panel B
    # mode: joint              # baseline, no transparent layer
```

**Panel A (`unified`)** — one shared graph over 53 Neuromark nodes. Every modality
projects onto it, and **modality is kept as a separate channel block on each
node**, which is the design choice that makes A worth building: an edge between a
node led by GM and a node led by fALFF *is* "this grey-matter pattern co-varies
with this fALFF pattern". A decoder reconstructs each modality from the shared
nodes, which is what makes A decomposition-like rather than merely deep
(`recon_weight: 0` ablates it).

**Panel B (`modality_specific`)** — one graph per modality over its own native
nodes: parcels for the structural streams, components for sFNC. No shared node
set, so nothing to project and nothing to justify. Cheaper, and per the proposal
"the most permissive to missing modalities".

**The node set is the claim.** sFNC needs no projection — it is already indexed by
component pairs, so component *i*'s row of the FNC matrix is node *i*'s feature
vector. The structural modalities do need one, and there are two honest options:
a prep-time overlap matrix (fixed node meaning, trustworthy names) or a learned
linear map (shared node set, but the name is a weaker claim). The model prints
which it used and refuses to let a learned projection pass as anatomy:

```
    GM             learned  LEARNED projection — node meaning for this modality
                            is fitted, so its name is a weaker claim
    sFNC           fnc      component-indexed, no projection needed
  !! at least one modality uses a learned projection.
```

**Non-bypassability is verified, not assumed.** p.4 requires the layer to be a
"critical, non-bypassable information bottleneck". `assert_non_bypassable()`
checks the task head's input width equals the graph's output width, so nothing is
concatenated in alongside it, and the test suite calls it for both panels.

**Reading the graph** is a separate command, because an adjacency nobody reads is
just an oddly-shaped MLP:

```bash
python -m src.readout --config conf/experiments/panel_a.yaml --folds 10 --top 20
```

It fits one model per fold and reports each edge as **present in k of n folds**,
because "large in the model I happened to look at" is not a finding. On the
synthetic fixture — pure noise, no real cross-modal structure — every edge comes
back 1/3 and every modality share sits near 25%, which is the correct answer and
the reason to trust the k-of-n column on real data.

## Decisions on record

| decided | what | revisit by |
|---|---|---|
| 2026-09-21 | **keep only subjects with sFNC** — cohort 2,526, not 3,525 | `data.cohort=all` |
| 2026-09-21 | **no dynamic fMRI for now** — static sFNC vector, not the p.6 biLSTM encoder | add `dfnc` to `data.modalities`; `prep/extract_dfnc.py` and the timecourse tooling are kept, not deleted |

Nothing for the dynamic path has been removed. `prep/extract_dfnc.py`,
`prep/find_timecourses.py` and `prep/survey_timecourses.py` all still work, and
what we learned about the obstacle is written down below so it does not have to
be rediscovered:

- run length is constant within **19 of 23 sites** and spans 140-240 TRs; TR is
  constant within **21 of 23**. So sequence length is close to a site label, and
  no feature-level correction touches it because it is the tensor's shape, not
  its values. Cropping to the shortest run (140) costs ~40% of the median (230),
  and does not equalise duration anyway since TR differs (140 x 2.0s vs 140 x 2.5s).
- `mdd_master_full.csv` already carries `TR`, `n_timepoints`, `ica53_tc` and
  `ica105_tc`, so most of this can be re-derived without touching the cluster.

## Options

All set in `conf/`, overridable on the command line (`key=value`).

### 1. Cohort — which subjects

```yaml
data:
  cohort: sfnc          # all | sfnc | super_clean
```

| value | n | meaning |
|---|---|---|
| `all` | 3,525 | everyone; volumes exist for all |
| `sfnc` | 2,526 | `in_clean` — subjects with ICA-53 sFNC. **CURRENT DEFAULT.** MDD 1,170 / HC 885 / BD 157 / SCZ 314 |
| `super_clean` | 2,426 | `in_super_clean` — subjects with ICA-105 |

The tiers are strictly nested (105 ⊂ 53 ⊂ all). Requiring sFNC costs 999 subjects
(490 MDD, 459 HC, 50 BD, 0 SCZ), so `all` keeps the most data but has no
connectivity for 28% of them — hence `data.allow_missing_modalities`.

### 2. Modalities

```yaml
data:
  modalities: [GM, WM, CSF, FALFF, sFNC]
  allow_missing_modalities: true    # train on subjects missing some streams
  representation: parcellated       # parcellated | voxelwise
```

| modality | source | dims |
|---|---|---|
| `GM` `WM` `CSF` | probseg volumes | 100 parcels, or ~239k voxels |
| `FALFF` `FALFF_globalC` | consortium voxelwise fALFF | same |
| `sFNC` | GIFT `fnc_corrs_all` (ICA-53) | 1,378 |
| `sFNC105` | ICA-105 | 5,460 |
| `dfnc` | sliding-window FNC (proposal p.4) | k states × 1,378 |
| `spectra` | GIFT `spectra_tc_all` | 129 × 53 |
| `TIV` | GM+WM+CSF volume | 1 (covariate) |

`allow_missing_modalities: true` uses a per-subject mask so a subject missing sFNC
still trains on their volumes, rather than being dropped.

### 3. Scanner / site

```yaml
site:
  mode: random_effect   # ignore | feature | combat | random_effect | adversarial
```

23 sites. Untreated, some of the signal is scanner rather than illness.

- `random_effect` — **proposal-aligned** (LME, pp. 16/24/32): site enters as a
  random intercept; fixed effects are estimated net of it
- `combat` — harmonise features across sites before modelling
- `feature` — site one-hot appended as an input (use when site *is* of interest)
- `adversarial` — encoder penalised for predicting site
- `ignore` — baseline, for measuring how much site actually matters

### 4. HAMD labels

```yaml
label:
  target: diagnosis     # diagnosis | hamd17 | hamd3
  hamd_mode: transfer   # direct | transfer | holdout
```

Counted from the real cohort — and the two scores are **not** interchangeable:

| | whole cohort | at `cohort: sfnc` |
|---|---|---|
| HAMD17 total | 1,394 | **984** |
| HAMD3 (item 3) | 989 | **660** |
| HAMD3 without HAMD17 | 0 | 0 |

HAMD3 is nested inside HAMD17, so a HAMD17 target has 984 subjects and a HAMD3
target has 660. Availability follows the target (`src/data.py: hamd_available`);
requiring both regardless used to cost 324 subjects on every HAMD17 run.

**HAMD3 is the suicide item**, and three sites recorded the total but never item 3
(IS001 254 subjects, IS008 47, IS018 34), plus three sites recorded no HAMD at all.
So the HAMD3 target loses whole sites, not just subjects — which is also why
`hamd_mode=holdout` is close to a site-transfer experiment here.

- `direct` — train and test on HAMD subjects only (n ≈ 613–660)
- `transfer` — **pretrain encoders on diagnosis using all subjects**, then
  fine-tune the head on the HAMD subset. Lets the 2,900 unlabelled subjects
  contribute.
- `holdout` — train on subjects *without* HAMD, predict on those *with*. Tests
  whether a diagnosis-trained representation carries symptom information it was
  never shown.

---

## Layout

```
conf/
  base.yaml                 defaults for everything
  experiments/*.yaml        one file per run, overrides base
src/
  config.py                 YAML + CLI override
  data.py                   cohort selection, modality loading, missing-modality masks
  site.py                   the five site strategies
  transparent.py            THE transparent layer: named nodes, learnable graph,
                            decoder, modality attribution
  models.py                 encoders; panel A (unified), panel B
                            (modality_specific), and the no-graph baselines
  readout.py                turns a trained graph into named, fold-stable edges
  train.py                  CV loop (kfold|loso), pretrain/fine-tune staging
  evaluate.py               AUC, LME on fold results, correlations
  diagnose.py               confound report — run this FIRST
prep/
  build_index.py            builds data/mdd_master.csv -- the spine everything joins on
  extract_struct.py         parcellate GM/WM/CSF/fALFF volumes -> features
  extract_sfnc.py           pull sFNC/spectra from the GIFT MATs (tier-aware join)
  extract_dfnc.py           sliding-window dFNC (proposal p.4)
  make_synthetic.py         fake cohort, so the pipeline runs without the server
tests/
  test_pipeline.py          the four failures that would otherwise be silent
scripts/
  run_prep.sh  run_train.sh  run_diagnose.sh  SLURM wrappers. Each is SELF-CONTAINED:
  run_find_tc.sh  run_survey_tc.sh            conda activation is inlined (it is
                                              disabled on the login node), and they
                                              locate their .py files whether the
                                              layout is flat or the repo tree.
```

## Quick start

Check it works before touching real data -- the synthetic cohort reproduces the
awkward parts (nested QC tiers, subjects missing sFNC, HAMD on a minority of MDD):

```bash
python tests/test_pipeline.py                   # 7 checks, no server needed
python prep/make_synthetic.py --n 400
python -m src.train --config conf/experiments/hc_vs_mdd_sfnc.yaml \
       train.n_folds=3 train.n_repeats=1 train.epochs=8
```

Then on the server. The scripts anchor on `SLURM_SUBMIT_DIR` -- the directory you
ran `sbatch` from -- so run them from wherever you keep them, or set
`REPO=/path/to/them`. `run_find_tc.sh` and `run_survey_tc.sh` work from a flat
directory; the rest import the `src` package and need the repo tree.

```bash
sbatch scripts/run_prep.sh                      # index + all feature files, once
sbatch scripts/run_train.sh conf/experiments/hc_vs_mdd_sfnc.yaml
sbatch scripts/run_train.sh conf/experiments/hamd_transfer.yaml site.mode=combat
sbatch scripts/run_sweep.sh                     # all four comparisons + the LME
python -m src.evaluate 'runs/*.json' --target diagnosis --factor site.mode
```

## Run the diagnostics before believing any result

```bash
python -m src.diagnose --config conf/experiments/hc_vs_mdd_sfnc.yaml
```

Four questions, all derived from our own data:

1. **The site-only baseline** — cross-validated AUC for predicting diagnosis from
   the one-hot site vector and nothing else. This is the floor every model AUC
   should be read against, not 0.5. If it comes back at 0.68 and the fusion model
   scores 0.72, the model added 0.04.
2. **Is HAMD availability a property of site?** If the ~660 HAMD subjects come
   from a few sites, `hamd_mode=holdout` is a site split in disguise — it trains
   on subjects without HAMD and tests on subjects with it.
3. **ICC of HAMD17 across sites** — how much severity variance is between sites
   rather than between people.
4. **Per-feature eta-squared by site** — which modality carries the loudest
   scanner signature.

**Then run leave-one-site-out**, because random k-fold structurally cannot detect
site leakage: the same scanner appears in train and test.

```bash
python -m src.train --config conf/experiments/loso.yaml
```

On a fixture with **zero** within-site diagnosis signal, where only the MDD rate
differs across sites, the repo's own test suite measures:

| | AUC |
|---|---|
| site-only baseline | 0.92 |
| random k-fold, all four modalities | 0.93 |
| leave-one-site-out | 0.46 |

The k-fold number lands on the site-only baseline — it learned the scanner and
nothing else — and would have been reported as a strong multimodal result. That
comparison is `tests/test_pipeline.py` check 5, so it stays honest.

**Also run `site_check.yaml`.** It predicts the *scanner* from the features
instead of the diagnosis. If that AUC is near 1.0 and diagnosis tracks site, every
other number in the repo needs harmonisation before it means anything.

**Read the fold SD, not just the mean.** Two identical fusion runs previously moved
0.023 apart on this data, which is why `train.n_repeats` defaults to 3 and why
`evaluate.py` prints the within-run SD next to every mean. A gap smaller than twice
that SD is not a finding. `--factor` tests it with the mixed model instead of by eye.

## Known limitations, stated rather than buried

- **dFNC state features are mildly optimistic.** `extract_dfnc.py` fits the k-means
  over windows across all subjects at once, so the clustering has seen the test
  fold. Occupancy and dwell inherit that. Edge variability does not. Treat dFNC
  results as exploratory until the clustering is refitted inside each training fold.
- **Schaefer-2018 is cortex-only.** No hippocampus, no amygdala -- a real gap for
  MDD. `extract_struct.py --subcortical harvard_oxford` appends subcortical parcels;
  `run_prep.sh` turns that on by default.
- **GM and WM are nearly redundant** (r = -0.91 on this data). CSF is not
  (r = -0.04 with GM), which is why the default modality set keeps CSF and drops WM.
- **`representation: voxelwise` is declared but not implemented.** The volumes are
  parcellated at prep time. Voxelwise needs in-fold PCA or a 3D CNN encoder, and
  doing it without in-fold dimensionality reduction would leak.
- **`holdout` gives one split, not folds.** The evaluation set is fixed by who has
  HAMD, so repeats vary only the initialisation, not the partition.
