# DIRECT-II: what is on the remote, and what sMRI actually exists

Status: 2026-09-03. Source: local mount of `MDD_DIRECT` + the three shipped
`*_filelist.txt` manifests (which index the archive contents without needing
to unzip 1.9 TB).

## 1. Archives on the remote

| Archive | Size | Extracted? | Contents |
|---|---|---|---|
| `DIRECT_II_Results.zip` | 112 GiB | no | `Results/Results/{AnatVolu,AnatSurfLH,AnatSurfRH,FunVolu,FunSurfLH,FunSurfRH,ROISignals_*}`, `Results/Masks`, `Results/RealignParameter`, `Results/TRInfo.tsv` |
| `FunVoluW_RP.zip` | 1.79 TB | yes → `Data_BIDS/` | volumetric BOLD time series + masks + realignment params |
| `FunSurfW_RP.zip` | 57 GiB | no | surface BOLD time series, `space-fsaverage5_hemi-{L,R}.func.gii` |

3525 subjects with imaging. IDs `IS<site>-<cohort>-<nnnn>`, 23 sites, 54
site-cohort folders. Note the cohort digit is not a clean MDD/NC flag —
`IS003-3`, `IS004-4`, `IS006-4`, `IS019-4` exist — so diagnosis must be derived
from membership in `MDD.xlsx` vs `NC.xlsx`, not from the ID.

## 2. Volumetric fMRI (extracted, usable now)

`Data_BIDS/FunVoluW/<ID>/` holds three 4D series per subject:

| File | Grid | Smoothing | Size |
|---|---|---|---|
| `<ID>_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii` | fMRIPrep native (~2 mm) | none | ~500 MB |
| `Np<ID>_...` | SPM12-normalised 3×3×3 mm, bbox `[-78 -112 -70; 78 76 85]` | none | ~80 MB |
| `SmNp<ID>_...` | same 3 mm | 6×6×6 mm FWHM | ~80 MB |

Prefixes decoded from `Data_BIDS/Code_Prep/Step2_prep_SPM.m`: `p` = resave,
`N` = normalise, `Sm` = smooth. No slice-timing correction was applied.
The `Np`/`SmNp` grid is the Neuromark input grid.

Supporting volumetric files:
- `Data_BIDS/Masks/AutoMasks/<ID>_..._desc-brain_mask.nii.gz` (3525)
- `Data_BIDS/Masks/SegmentationMasks/MNIFunSpace_<ID>_{WM,CSF}.nii` (14100 = 2×3525, plus group-level masks)
- group masks `AllResampled_{Brain,Grey,White,Csf}Mask_*_91x109x91.nii`

## 3. fALFF: compute on `Np`, smooth afterwards

DPARSF/DPABI convention is to compute ALFF/fALFF on unsmoothed normalised data
and smooth the resulting map, the reverse of the FC convention. DIRECT followed
this: their own `Results/Results/FunVolu/fALFF_FunVoluWC/fALFF_<ID>.nii.gz`
files carry no `s` (smooth) or `z` (z-score) prefix, i.e. they are the raw
unsmoothed fALFF maps.

Recommended chain for our own fALFF: `Np` → bandpass + fALFF → optional 6 mm
FWHM smooth → optional z-score within brain mask.

DIRECT ships both `C` (nuisance-regressed) and `globalC` (+ global signal
regression) variants of each measure. We are recomputing, so this only matters
if we want to reproduce their numbers as a sanity check.

## 4. sMRI: no T1 volume was ever released

But we do have derivatives like tissue probability maps, we may use them.
## 5. Metadata

- `Data_Info/TRInfo.tsv` — TR, slice count, timepoints, voxel size per subject
  (duplicated at `Data_BIDS/TRInfo.tsv`)
- `Data_Info/MDD.xlsx` — 2498 rows × 74 cols (60 named). Full column list:
  `ID, Sex, Age, Education, AO, IllnessDuration, FirstEpisode, Epi,
  CurrentEpisodeDuration, EverMedication, OnMedication, MedicationName,
  MedicationDose, TreatmentResponsive, FamilyHistory, Other, HAMDTotal17,
  HAMD1..HAMD24, HAMATotal, HAMA1..HAMA14, CGIS, Other2, RRS, WSCT`.
  **HAMD3 is the suicide item.** Note HAMD goes to 24 items (not 17) and there
  is a full HAMA anxiety scale, CGI-S, Ruminative Response Scale and Wisconsin
  Card Sorting — richer than the Phase I release.
- `Data_Info/NC.xlsx` — 1341 rows × 7 cols (`ID, Sex, Age, Education, Other, RRS, WSCT`)
- `Data_Info/Legend.xlsx` — Chinese→English codebook; also documents optional
  bipolar and schizophrenia tables that this release does not populate
- `Data_BIDS/RealignParameter/HeadMotion.tsv` — 21 columns incl. `mean FD_Power`,
  `mean FD_Jenkinson`, `Percent of FD_Power>0.2`
- `Data_BIDS/RealignParameter/ExcludeSubjectsAccordingToMaxHeadMotion.txt` —
  exclusion list at 3.0 mm / 3.0 degree max head motion

Demographic coverage (3839 rows across the two sheets) exceeds imaging coverage
(3525 subjects), so the manifest is driven by the imaging directory listing and
left-joins the clinical tables.


