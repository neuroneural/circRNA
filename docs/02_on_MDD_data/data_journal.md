# MDD DIRECT Data Investigation Notes

Notes on the depression dataset living under the remote `phase_prediction` codebase
(`~/_remote_data/phase_prediction`). Goal: understand what is there and how to reuse
it (labels + timeseries) in a separate analysis.

## 1. Overview

* **What it is**: The **MDD DIRECT** cohort = a subset of the **Depression Imaging REsearch ConsorTium (DIRECT) / REST-meta-MDD** project — a large **multi-site Chinese resting-state fMRI** study of Major Depressive Disorder (25 teams, ~17 hospitals in China).
* **Subject counts**:

| File | Subjects | Role |
| :--- | :---: | :--- |
| `MDD_DIRECT_FULL.csv` | 3,525 | Full demographic + clinical table (master `MATLABIndex` 1–3525) |
| `MDD_DIRECT_TR2.0_crop230_subset.csv` | 2,426 | Working subset (TR=2.0 s, cropped to 230 timepoints) — matches the local `.pt` files |
| `MDD_China_organized.csv` | 600 | Separate smaller table with its own ID + group scheme |

* **Files that matter most** (everything else is derived from these):

| File(s) | What it gives you |
| :--- | :--- |
| `data/MDD_DIRECT_*.csv` | Labels + demographics + clinical scores, keyed by `Subject` / `MATLABIndex` |
| `data/converted_files_mddd_*.csv` | Per-representation manifest: maps `Subject` → original file → converted `.pt` file |
| `NeuroFLAME_shapes_all.csv` | Filename + array shape of every ICA-timecourse NIfTI (useful for scan lengths) |
| `phase_prediction/dataloaders/CSVDataset.py` | Reference loader (how `.pt` + label are joined at train time) |

### 1.1 The `MDD_preproc` project (ICA source + subject selection)

`~/_remote_data/MDD_preproc` (cluster: `/data/users3/bbaker/projects/MDD_preproc/`) is the **upstream project** that produced every feature used in `phase_prediction`. Two things there are directly relevant to us: **the ICA data** and **the definition of the "clean" subset**.

* **ICA decompositions** exist in two resolutions, each with QC variants — 53-component (`neuromark53_clean/`) and 105-component / **Neuromark 2.2** (`results/ica/neuromark105_clean/`, `neuromark105_super_clean/`). Each run yields timecourses, spatial maps, sFNC, and spectra (see §3.1).
* **Subject QC / selection tiers** — this is what "clean subset" means:

| Tier | List file (`data/MDDD/`) | n | Selection |
| :--- | :--- | :---: | :--- |
| full | `Subjects.txt` | 3,525 | all DIRECT subjects |
| clean | `Subjlist_clean.txt` | 2,526 | QC-passed (usable ICA) |
| **super_clean** | `Subjlist_super_clean.txt` | **2,426** | clean **∩ TR = 2.0 s, cropped to 230 timepoints** — this is the `TR2.0_crop230` working set used everywhere downstream |

* The subject-list files point at the **raw preprocessed BOLD** (fMRIPrep-style, MNI152NLin2009cAsym): `/data/qneuromark/Data/Depression/MDD_DIRECT/Data_BIDS/FunVoluW/{ID}/SmNp{ID}_task-rest_..._desc-preproc_bold.nii`. Per-subject TR is listed in `Subjlist_super_clean_TR.txt` (all `2.0` for super_clean).
* **`data/MDDD/FULL_sites.csv`** = the same clinical table as `MDD_DIRECT_FULL.csv` **plus a `Site` column** (drives their ComBat / leave-one-site-out harmonization). Keyed by `ID` (= `Subject`) + `MATLABIndex`.
* Caveat: exact motion/quality thresholds for "clean" are **not** spelled out in the Python; the selection is baked into the `Subjlist_*.txt` lists and `TR2.0_crop230_indices.mat`.

### 1.2 The `MDD_DIRECT` raw-data root (data dictionary + processed volumes)

`~/_remote_data/MDD_DIRECT` (cluster: `/data/qneuromark/Data/Depression/MDD_DIRECT/`) holds the **consortium metadata** and the **processed fMRI volumes**, upstream of the ICA in §1.1.

* **`Data_Info/` — consortium codebook & source demographics** (answers "how are labels defined / is there a dictionary"):
  * `Legend.xlsx` — authoritative **codebook** (Chinese→English column names + coding rules, e.g. `Sex` 1=M/2=F, `Education` in years, `FirstEpisode` 1/0, `TreatmentResponsive` 1=non-refractory/0=refractory). Has **four sheets: MDD, healthy control, bipolar (optional), schizophrenia (optional)**.
  * `MDD.xlsx` / `NC.xlsx` — original per-subject demographics for patients / controls.
  * `TRInfo.tsv` — per-subject acquisition: TR, slices, **time points (232)**, voxel size. Basis for the `TR2.0_crop230` selection.
* **`Data_BIDS/`** — working tree: `FunVoluW/` (the processed volumes, below), `Masks`, `RealignParameter/` (head-motion params, for motion QC), `Code_Prep`, `TRInfo.tsv`.

**Processed volume fMRI** — one folder per subject at:

* `~/_remote_data/MDD_DIRECT/Data_BIDS/FunVoluW/{ID}/`
* cluster: `/data/qneuromark/Data/Depression/MDD_DIRECT/Data_BIDS/FunVoluW/{ID}/`

Each folder has **three** 4-D BOLD NIfTIs — all `task-rest`, `space-MNI152NLin2009cAsym`, `desc-preproc`, `int16`, 232 timepoints (TR 2 s). They differ only in grid/smoothing:

| File (prefix) | Grid | Voxel | What it is | Use if… |
| :--- | :--- | :--- | :--- | :--- |
| `{ID}_…_preproc_bold.nii` (no prefix, ~500 MB) | 97×115×97 | 2 mm | full-res fMRIPrep output in MNI 2 mm | you want high-res standard-space voxel data for your own pipeline |
| `Np{ID}_…_preproc_bold.nii` (~80 MB) | 53×63×52 | 3 mm | resampled to the **Neuromark** analysis grid, **un-smoothed** | you want the analysis grid but your own smoothing |
| `SmNp{ID}_…_preproc_bold.nii` (~80 MB) | 53×63×52 | 3 mm | **Sm**oothed + resampled (Neuromark grid) | **you want the exact input the ICA/FNC used** ← default |

BIDS-name decode: `{ID}` = subject `IS{site}-{group}-{num}` (group = 1 MDD / 2 control / 3 bipolar / 4 schizophrenia, see table below) · `task-rest` = resting state · `space-MNI152NLin2009cAsym` = normalized to that MNI template · `desc-preproc` = preprocessed · `bold` = functional image · `.nii` = uncompressed NIfTI. The `Np` / `SmNp` prefixes are pipeline-added (Neuromark-grid resample; `Sm` = spatial smoothing). The `SmNp` file is the one listed in `Subjlist_super_clean.txt`, i.e. the ICA input.

* **Subject-ID scheme encodes diagnostic group** (middle digit), verified against `Diagnosis`:

| ID group | Group | n | `Diagnosis` |
| :--- | :--- | :---: | :---: |
| `IS…-1-…` | MDD | 1,660 | `1.0` |
| `IS…-2-…` | Healthy control | 1,344 | `0.0` |
| `IS…-3-…` | Bipolar | 207 | *(blank)* |
| `IS…-4-…` | Schizophrenia | 314 | *(blank)* |

  So the **blank `Diagnosis` rows are the bipolar + schizophrenia subjects** (521 total), not missing data — they're simply outside the MDD-vs-control contrast, but their imaging is available here for transdiagnostic use.

## 2. Metadata CSVs & Labels

### 2.1 How the MDD label is assigned

* The label is the **`Diagnosis`** column in the `MDD_DIRECT_*.csv` tables:

| `Diagnosis` | Meaning | n (subset) |
| :---: | :--- | :---: |
| `1.0` | **MDD patient** | 1,123 |
| `0.0` | **Healthy control** | 831 |
| *(blank)* | Unlabeled / not usable | 472 |

* **Basis of the label**: it is a **clinical group assignment** made at the recruiting hospital under **DSM-IV (or ICD-10) criteria for MDD** — the DIRECT consortium standard. It is **not** derived from a HAMD threshold in this dataset: controls carry no HAMD score at all, and 186 labeled patients (1,123 − 937) have no HAMD total, yet are still labeled `1.0`. (Some downstream consortium analyses additionally require HAMD-17 ≥ 8 as a *severity* filter, but that is a filter, not the label definition.)
* **No data-dictionary PDF exists or is needed** — the coding is verifiable directly from the clinical columns (they are populated only for `Diagnosis = 1.0`).
* **Caveat — the `MDD_China_organized.csv` file uses a different scheme**: its `group` column is `1` = patient (HDRS populated, n=284) and `2` = control (no HDRS, n=316). Do not assume the same coding as `Diagnosis`.

### 2.2 Clinical scales: HAMD vs HAMA

Both Hamilton scales are present. **HAMD measures depression; HAMA measures anxiety** — they are distinct instruments:

| Scale | Full name | Measures | Columns | Availability (subset) | Observed range (total) |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **HAMD** | Hamilton **Depression** Rating Scale | General depression **severity** | `HAMDTotal17`, `HAMD1`–`HAMD24` | Total: 937 · items 1–17: ~613 · items 18–24: ~289 | 0–52, mean ≈ 21 (moderate–severe) |
| **HAMA** | Hamilton **Anxiety** Rating Scale | Anxiety severity | `HAMATotal`, `HAMA1`–`HAMA14` | Total: 670 · items: ~360 | 0–47, mean ≈ 18 |

Notes on HAMD:

* **`HAMDTotal17`** is the standard **17-item** total (the main depression-severity number). `HAMD18`–`HAMD24` are the **24-item extension** items (available for fewer subjects); individual items are scored 0–4.
* HAMD is a **depression** scale, but it embeds **two anxiety items** — `HAMD10` (psychic anxiety) and `HAMD11` (somatic anxiety). If you need a clean anxiety measure, use **HAMA**, not those sub-items.
* So: **use `HAMDTotal17` for depression severity, `HAMATotal` for anxiety severity.**

Other clinical columns: `RRS` (Ruminative Response Scale, n=62), `Age`, `Sex`, `Education`, `AO` (age of onset), `IllnessDuration`, `FirstEpisode`, `OnMedication`/`EverMedication`, `TreatmentResponsive`, `FamilyHistory`. `CGIS` and `WSCT` columns are present but empty.

### 2.3 Suicidality rating (HAMD item 3)

The only suicide-specific measure in the data is **HAMD item 3 (`HAMD3`)** — the HAMD "Suicide" item (suicidal ideation/attempt), scored **0–4** (0 = absent → 4 = most severe).

* **Patients only.** `HAMD3` is populated for **613 subjects, all MDD** (`Diagnosis = 1.0`); **no HC, SZ, or BP** have it. In fact *no* HAMD or HAMA score exists for any healthy control — these scales were administered only to the depressed group.
* **Coverage** matches the per-item HAMD (613), i.e. a subset of the 937 subjects who have a `HAMDTotal17`. Distribution in `crop230`: 0→217, 1→166, 2→84, 3→103, 4→43.
* `HAMDTotal17` includes this item but is not suicide-specific; use `HAMD3` directly for a suicidality measure.

### 2.4 Gotchas

* Some fields contain **Chinese text** (e.g. `MedicationName` = `无` meaning "none").
* In a few converted manifests the `age` column is stored as a **string like `tensor([[26.]])`** — parse the number out before use.

## 3. Neuroimaging Data (modalities & shapes)

Three levels, from raw to most-derived. **Cluster paths** are given below (the same trees are mounted locally under `~/_remote_data/`).

| Modality | Format | Per-subject shape | Cluster path |
| :--- | :--- | :--- | :--- |
| **Volume / 4D fMRI** | NIfTI (`…_desc-preproc_bold.nii`) | `97×115×97` @2 mm (base) or `53×63×52` @3 mm (`Np`/`SmNp`), 232 TRs | `/data/qneuromark/Data/Depression/MDD_DIRECT/Data_BIDS/FunVoluW/{ID}/` — see §1.2 |
| **ICA timecourses** | NIfTI (`..._timecourses_ica_s1_.nii`) | `[T × C]`, T ≈ 230–232, **C = 53 or 105** components | `/data/users3/bbaker/projects/MDD_preproc/neuromark53_clean/` (53) · `/data/users3/bbaker/projects/MDD_preproc/results/ica/neuromark105_super_clean/` (105) — see §3.1 |
| **FNC (static & dynamic)** | PyTorch `.pt` (one tensor/subject) | see below | `/data/users3/bbaker/projects/phase_prediction/data/mddd_data_*/` |

### 3.1 ICA results — timecourses (primary) + what else the decomposition carries

Spatial ICA decomposes each subject's 4-D BOLD into **spatial maps** and their **timecourses**. The connectivity/amplitude summaries (**sFNC**, **spectra**, component **fALFF**) are then computed **from the timecourses**. Contents and shapes below are exactly what `scripts/assemble_mddd_data.py` extracts:

| Modality | Where it lives | Key / filename pattern | Shape | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Timecourses (TC)** | NIfTI in the ICA dir | `MDD_DIRECT_sub{NNN}_timecourses_ica_s1_.nii` | `(T, N_comp)` | ICA **co-product** — component timeseries (already ICA-reduced, not voxelwise); primary target for timeseries work; `nibabel`-loadable; T ≈ 230, N_comp = 53 or 105 |
| **Spatial maps** | NIfTI in the ICA dir | `MDD_DIRECT_sub{NNN}_component_ica_s1_.nii` | `(X, Y, Z, N_comp)` | ICA **co-product**, produced *jointly* with the TCs (not derived from them) — component spatial patterns |
| **sFNC** (static FNC) | postprocess `.mat` | `fnc_corrs_all` | `(S, N_comp, N_comp)` | *derived from TCs*: **Fisher-z** component×component correlation |
| **Spectra** | postprocess `.mat` | `spectra_tc_all` | `(S, F, N_comp)` | *derived from TCs*: power spectra of the timecourses |

(S = subjects, F = frequency bins.)

**fALFF** (fractional Amplitude of Low-Frequency Fluctuations) = the fraction of a signal's total spectral power that falls in the low-frequency band (~0.01–0.08 Hz) — i.e. the relative strength of slow spontaneous fluctuations. Classically it is a **voxelwise 3-D map** (one value per voxel). Two forms exist here:

* **Component-level fALFF** (what this ICA pipeline uses — hence the Neuromark link): fALFF of each ICA component's *timecourse* → **one value per component (53/105) per subject**, derived from the TCs like sFNC/spectra. Path: `/data/users3/bbaker/projects/MDD_preproc/results/falff/` — `falff_all_super_clean.mat` (≈ subjects × 105) plus per-subject `cmdd/subj{NNN}_falff.mat`; the `drange_*` files are matching dynamic-range measures.
* **Voxelwise / surface fALFF** (the classic 3-D form): produced by the DPABI pipeline and stored inside `MDD_DIRECT/DIRECT_II_Results.zip` under `Results/Results/FunVolu/fALFF_FunVoluW*` (volume) and `FunSurfLH|RH/fALFF_FunSurfW*` (surface).

The same ICA run is stored in three forms:

* **Exported per-subject NIfTI** — the TC and spatial-map files above (e.g. `neuromark53_clean/`, ~49 KB per TC). This is the cleanest entry point if you just want timeseries.
* **GIFT back-reconstruction** — `MDD_DIRECT_ica_br{N}.mat` (one per subject; TC + maps), plus the group `MDD_DIRECT_ica.mat`, `MDD_DIRECTMask.nii`, `MDD_DIRECTSubject.mat`, and `MDD_DIRECT_gica_results/`.
* **Consolidated postprocess `.mat`** — `MDD_DIRECT_postprocess_results.mat` holds `fnc_corrs_all` + `spectra_tc_all` for all subjects (load with `mat73`); ≈ 185 MB (53-comp) / ≈ 0.7 GB (105-comp).

**Two resolutions**: 53-component (`neuromark53_clean/`) and 105-component / Neuromark 2.2 (`results/ica/neuromark105_clean/`, `neuromark105_super_clean/`).

**Original / full ICA output** (unfiltered 3,525-subject GIFT results incl. `_br`/`_c` intermediates, numbered `sub{N}` 1–3525) lives at `/data/qneuromark/Results/ICA/MDD_DIRECT/{Neuromark1.0 = 53, Neuromark2.2 = 105, NeuromarkWM}/` — the source that the `*_clean` copies are filtered/renumbered from.


### 3.2 dFNC — dynamic (sliding-window) FNC

* **dFNC gives many FNC matrices per ICA timecourse** — one per sliding window. Each subject's file is a stack of windowed connectivity vectors.
* Shape = **`[n_windows × 1378]`**, where `1378 = 53 × 52 / 2` = the **upper triangle** of the 53×53 component FNC matrix. dtype float32.
* Window count follows **`n_windows = 230 − (window_seconds / 2)`** (TR = 2 s, step = 1 TR, 230-timepoint crop). Verified on `sub_1`:

| Folder | Window (s) | Window (TRs) | n_windows |
| :--- | :---: | :---: | :---: |
| `mddd_data_dfnc_20s` | 20 | 10 | 220 |
| `mddd_data_dfnc_40s` | 40 | 20 | 210 |
| `mddd_data_dfnc_60s` | 60 | 30 | 200 |
| `mddd_data_dfnc_88s` | 88 | 44 | 186 |
| `mddd_data_dfnc_100s` | 100 | 50 | 180 |

  (A `mddd_data_dfnc_10s` folder exists but appears **incomplete** — `sub_1` is missing.)

### 3.3 sFNC — static FNC

* One connectivity matrix per subject over the whole scan (no time axis).
* 105-component version (`mdd_direct_ica105_sfnc`): a **5,460-length** vector = upper triangle of the 105×105 matrix.
* Also derived and local for the 105-component ICA: **`spectra`** and **`mips`** (`mdd_direct_ica105_spectra`, `mdd_direct_ica105_mips`).

### 3.4 Voxelwise maps (fALFF / ALFF / ReHo) — archive only

* **Where on the cluster**: `/data/qneuromark/Data/Depression/MDD_DIRECT/DIRECT_II_Results.zip` (~112 GB; contents listed in `DIRECT_II_Results_filelist.txt`).
* **Where inside the zip** (per-subject 3-D NIfTI): `Results/Results/FunVolu/fALFF_FunVoluW{C,globalC}/fALFF_{ID}.nii.gz` — **3,525 subjects × 2 variants = 7,050 files**.
* **Two variants (the 2 file types)**:
  * `fALFF_FunVoluWC/` — nuisance-regressed, **without** global-signal regression
  * `fALFF_FunVoluWglobalC/` — same, **with** global-signal regression
* Extract just what you need, e.g.: `7z x DIRECT_II_Results.zip "Results/Results/FunVolu/fALFF_FunVoluWC/*" -o<dest>`.
* Not to be confused with `MDD_preproc/results/falff/` — that is *component-level* fALFF (one value per Neuromark component), see §3.1.

## 4. Mapping Metadata → Neuroimaging Data

### 4.1 Join keys

| Key | Example | Use |
| :--- | :--- | :--- |
| **`Subject`** | `IS001-1-0001` | **Primary join key** — present in every metadata CSV *and* every `converted_files_*` manifest. Use this to join anything. |
| **`MATLABIndex`** | `1 … 3525` | Row index into the **full** demographics table (`MDD_DIRECT_FULL` / `FULL_sites`). It is **NOT** the ICA file index — see §4.4. |
| **list position** | `1 … 2526` (clean) / `1 … 2426` (super_clean) | 1-based position of the subject in an ICA folder's subject list. This is how the per-subject ICA NIfTIs are numbered and how the postprocess sFNC/spectra rows are ordered. |

### 4.2 The manifests

Each representation has a `data/converted_files_mddd_<rep>.csv` with columns
`old_filename` (original source path), `new_filename` (converted `.pt` path), `Subject`, `session`, and `score`/`age`. This is what turns a `Subject` into a concrete file for that modality.

### 4.3 Recipe

1. **Start from a metadata CSV** (`MDD_DIRECT_TR2.0_crop230_subset.csv`) for labels/demographics, keyed by `Subject`.
2. **For a derived feature** (dFNC / sFNC): join to `converted_files_mddd_<rep>.csv` on `Subject`; read the file at `new_filename`.
3. **For ICA timecourses / spatial maps**: the file index is the subject's **1-based position in that ICA folder's subject list** — `Subjlist_clean.txt` for `neuromark53_clean/`, `Subjlist_super_clean.txt` for `neuromark105_super_clean/` — i.e. `MDD_DIRECT_sub{pos:03d}_…_ica_s1_.nii`. The postprocess sFNC/spectra arrays use that **same position** as their row index. (For the raw volume, index by `Subject` directly — the folder is named by ID.)

### 4.4 Things to fix / watch

* **Path repointing**: `new_filename` in the manifests uses the cluster prefix
  `/data/users3/bbaker/projects/phase_prediction/data/<subdir>/…`.
  The `<subdir>` names match the local `data/` exactly, so a prefix
  string-replace is enough to use the files locally.
* **ICA file numbering — do NOT use `MATLABIndex`.** The per-subject ICA NIfTIs are numbered by **position in the folder's subject list**, not by `MATLABIndex`:
  * `neuromark53_clean/` → `Subjlist_clean.txt` order (`sub001…sub2526`)
  * `results/ica/neuromark105_super_clean/` → `Subjlist_super_clean.txt` order (`sub001…sub2426`)

  Verified example: `IS015-1-0001` has `MATLABIndex = 2645`, but its files are `sub2037` (53, clean position) and `sub2034` (105, super_clean position) — `sub2645` does not exist. (An older `neuromark{53,105}/` *source* folder referenced by `phase_prediction`'s convert scripts **was** MATLABIndex-numbered; don't confuse it with these clean folders.)
* **`phase_prediction` `.pt` numbering** is different again: local dFNC files are `sub_1…sub_2426` in subset order. **Always join on `Subject`, never on any file number.** `build_mdd_master_csv.py` resolves all three numbering schemes for you (`clean_row` / `super_clean_row`).

---
*Date:* 2026-07-17
*Author:* Pavel Popov
