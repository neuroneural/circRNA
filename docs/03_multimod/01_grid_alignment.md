# Grid and alignment across MDD DIRECT volumetric modalities

Headers read from `IS001-1-0001` … `IS001-1-0004`. **The geometry below is identical across
all four subjects for each modality** — dims, voxel size, orientation, sform/qform codes and
affine. 
| Modality | Space | Dims | Vox (mm) | Orient | sform | Affine offset | World bbox x / y / z | dtype |
|---|---|---|---|---|---|---|---|---|
| `Np` bold | MNI152NLin2009cAsym † | 53×63×52×T | 3 | **LAS** | **2** | (78, −112, −70) | −78…78 / −112…74 / −70…83 | int16 |
| `SmNp` bold | MNI152NLin2009cAsym † | 53×63×52×T | 3 | **LAS** | **2** | (78, −112, −70) | −78…78 / −112…74 / −70…83 | int16 |
| native bold | MNI152NLin2009cAsym | 97×115×97×T | 2 | RAS | 4 | (−96.5, −132.5, −78.5) | −96.5…95.5 / −132.5…95.5 / −78.5…113.5 | int16 |
| `fALFF` | MNI152NLin2009cAsym ‡ | 97×115×97 | 2 | **LAS** | 4 | (96.5, −132.5, −78.5) | −95.5…96.5 / −132.5…95.5 / −78.5…113.5 | float32 |
| `fALFF_globalC` | MNI152NLin2009cAsym ‡ | 97×115×97 | 2 | **LAS** | 4 | (96.5, −132.5, −78.5) | −95.5…96.5 / −132.5…95.5 / −78.5…113.5 | float32 |
| GM / WM / CSF probseg | MNI152NLin2009cAsym | 193×229×193 | 1 | RAS | 4 | (−96, −132, −78) | −96…96 / −132…96 / −78…114 | float32 |

† declared in the filename but
`sform_code` says 2 (`ALIGNED_ANAT`) — SPM resampled onto its own default bbox
(`[-78 -112 -70; 78 76 85]` in `Step2_prep_SPM.m`, normalise-write with no structural, so a
resample rather than a fresh estimate) and did not relabel the space. 

‡ the fALFF filenames
declare no space at all; it is inherited from the FunVoluW volumes DPABI computed them from.

All affines are diagonal — pure scale + translation, no rotation or shear anywhere.

## Details

**Three distinct sampling lattices, and no two of them coincide.** Not one pair can be
resampled onto another by pure index arithmetic; every combination interpolates.

**fALFF is the exact mirror of the native BOLD grid about x = 0.** Same dims, same 2 mm
spacing, same y and z — but the x voxel centres interleave: native sits at
−96.5, −94.5, … 95.5 and fALFF at −95.5, −93.5, … 96.5. The two sets are disjoint, offset
by 1 mm. Honouring the affine gives correct anatomy; reading array order and assuming RAS
silently left-right mirrors it.

**The 1 mm probsegs are not a clean 2× parent of the 2 mm grids.** probseg centres are on
integers, the 2 mm centres on half-integers, so downsampling by 2 does not land on them.

**`Np`/`SmNp` carry sform/qform code 2 (`ALIGNED_ANAT`), not 4 (`MNI_152`)** — SPM did not
label its own output as standard space. Tools that gate on the code will refuse to treat
these as MNI even though they are.

**DIRECT's fALFF maps are pre-masked with a fixed group mask.** All four subjects have
exactly 238,955 nonzero voxels. The probsegs are not masked — their nonzero counts vary per
subject (2,007,571 … 2,015,204). Any fALFF we compute ourselves will not carry that mask.

**BOLD is int16 with per-subject `scl_slope`/`scl_inter`** (e.g. 0.0304/838.81,
0.0331/892.18, …). Scaling must be applied. All derivatives are float32 with slope 1,
intercept 0.


