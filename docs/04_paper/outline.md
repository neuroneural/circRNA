# DIRECT-II rs-fMRI: what ML can and can't predict in MDD


Purpose:  see if MDD classification task can be potentially used as a pretraining objective useful for prediction of other useful categories: suicidality, medication, first/repeated episodes of depression. 
Motivation: improving accuracy on more difficult tasks is a hard problem, so using a related task for pretraining is a valid way to both improve scores and reveal hidden connections between the tasks that share common features
What we do in the process:  
- benchmark models on MDD classification task on DIRECT data, see what results they achieve on that task
- transfer them on other tasks and test different transfer configuration performance (frozen, encoder transfer and retraining, of full transfer and retraining)

**Data**
- DIRECT-II super_clean: 1,944 subjects (1,113 MDD / 831 HC), 14 sites
- NeuroMark ICA time courses: ICA-53 (main) and ICA-105
- ml4fmri models; within a task, every model uses the same 10 CV folds

## 1. MDD vs HC
- 13 ml4fmri models on ICA-53: meanMLP best, AUC 0.741 ± 0.046
- FNC models ~0.70 (BNT, BrainNetCNN, FBNetGen); DGCNN (Zhu et al. 2023) 0.645
- ICA-105: meanMLP 0.735, LR 0.724
- Context: REST-meta-MDD papers report 61–82% accuracy; site alone gives AUC ~0.59

TODO in the end of each results section, please lead with a short opening sentense telling what this experiment is (e.g., benchmarking models on MDD classification...), and close with what its main conclustion is (e.g., models are able to classify it, best models achieve this, plateau is this). no need to write a lot of results as a text, figures should be the showing point

## 2. Suicidality (HAMD item 3)
- Patients only (n = 613): meanMLP AUC 0.69 (0–1 vs 2–4) to 0.78 (0 vs 4)
- With HC scored 0: up to 0.79, but partly re-learns MDD vs HC
- All five scores as classes: meanMLP AUC 0.65 (patients), 0.73 (with HC)
- At the default threshold meanMLP almost always predicts the low class (none of the 43 scoring 4 flagged)
- To add: site-only baseline

## 3. Transfer
- Pretrain meanMLP on a pretraining task (770 subjects), then retrain all of it on a target task; no subject in both tasks
- Pretraining task: MDD vs HC, or sex as a control; subjects from separate sites or from shared sites
- Pretrained models learn their task: validation AUC 0.68 / 0.75 (MDD vs HC), 0.89 / 0.90 (sex), separate / shared sites
- Target task (the 612 HAMD-3 patients, two halves): HAMD-3 0–1 vs 2–4, first episode (582), on medication (449), sex
- MDD pretraining helps no target: ΔAUC −0.05 to +0.02 vs scratch, none significant
- Sex → sex transfers (+0.05 to +0.08 with shared sites), so the setup can detect transfer
- Frozen pretrained models (no retraining): MDD model 0.45–0.57 on the clinical targets and 0.50 on sex; sex model 0.73–0.77 on sex
- To add: site-only baselines (medication 0.82 and first episode 0.72 from scratch may be site)

## Bonus
- HAMD-17 regression (n = 934)

**Files**
- `paper_results.ipynb`: part A gathers finished runs; part B runs the transfer experiments
- `results/`: tables and figures
