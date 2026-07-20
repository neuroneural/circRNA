# MDD master CSV — column legend

Columns in **`mdd_master.csv`** (the lean/curated version). `mdd_master_full.csv`
has all of these **plus** every clinical column from `FULL_sites.csv` (see the
consortium codebook `MDD_DIRECT/Data_Info/Legend.xlsx` for those).

One row per subject. Blank = not available for that subject.

| Column | Description | Values / mapping |
| :--- | :--- | :--- |
| `id` | Subject ID (primary key) | `IS{site}-{group}-{num}`, e.g. `IS001-1-0192` |
| `site` | Acquisition site (from ID) | e.g. `IS001` |
| `group_code` | Diagnostic group (from ID middle digit) | `1`=MDD, `2`=HC, `3`=Bipolar, `4`=Schizophrenia |
| `group_label` | Text label of `group_code` | `MDD` / `HC` / `Bipolar` / `Schizophrenia` |
| `matlab_index` | Row index in the full (3,525) table | `1…3525`. **Not** the ICA file index |
| `Diagnosis` | MDD-vs-control label | `1.0`=MDD, `0.0`=HC, blank=SZ/BP (unlabeled) |
| `Sex` | Sex | `1`=male, `2`=female |
| `Age` | Age at MRI (years) | |
| `Education` | Education (years) | 0=illiterate, 6=primary, 9=junior, 12=high, 16=college, 18=grad+ |
| `HAMDTotal17` | Hamilton **Depression** total (17-item) | 0–52; **patients only** |
| `HAMATotal` | Hamilton **Anxiety** total | **patients only** |
| `HAMD3` | HAMD item 3 = **suicide** (ideation/attempt) | 0–4 (0=absent → 4=severe); **patients only** |
| `TR` | Repetition time (s) | e.g. `2`, `2.5` |
| `n_slices` | Slices per volume | |
| `n_timepoints` | Volumes in the scan | e.g. `232` |
| `voxel_size` | Acquisition voxel size (mm) | e.g. `2 2 2` |
| `in_clean` | In the `clean` QC tier (2,526) | `True`/`False` |
| `in_super_clean` | In the `super_clean` tier (2,426, TR=2 s) | `True`/`False` |
| `clean_row` | 0-based row into the **53-comp** postproc sFNC/spectra arrays | blank if not in `clean` |
| `super_clean_row` | 0-based row into the **105-comp** postproc sFNC/spectra arrays | blank if not in `super_clean` |
| `vol_np_3mm` | Path: 3 mm volume, resampled, **un-smoothed** | Neuromark grid 53×63×52 |
| `vol_smnp_3mm` | Path: 3 mm volume, resampled + **smoothed** | the ICA input |
| `ica53_tc` | Path: ICA-53 **timecourses** NIfTI | `(T, 53)` |
| `ica53_sm` | Path: ICA-53 **spatial maps** NIfTI | `(X,Y,Z,53)` |
| `ica105_tc` | Path: ICA-105 **timecourses** NIfTI | `(T, 105)` |
| `ica105_sm` | Path: ICA-105 **spatial maps** NIfTI | `(X,Y,Z,105)` |
| `sfnc_spectra_mat_53` | Path: 53-comp postproc `.mat` (sFNC + spectra) | index rows with `clean_row` |
| `sfnc_spectra_mat_105` | Path: 105-comp postproc `.mat` (sFNC + spectra) | index rows with `super_clean_row` |

*Notes:* HAMD/HAMA scores exist for MDD patients only (no HC/SZ/BP). ICA files are numbered by subject-list position, not `matlab_index` — join everything on `id`.
