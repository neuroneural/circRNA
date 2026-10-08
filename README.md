# MDD Data Analysis

This repository contains the analysis pipelines and exploratory logs for MDD data.

Currently, fold_data.py is a library of auc data for separate runs and a figure can be made by running figures/make_figs.p, which calls the fold data library.

To recreate the aucs contained in fold_data.py  you would run this

cd ~/...../mdd-multimodal

# --- all four modalities (blue boxes)

sbatch scripts/run_train.sh conf/experiments/baseline_blend.yaml

sbatch scripts/run_train.sh conf/experiments/meanmlp.yaml

sbatch scripts/run_train.sh conf/experiments/brainnetcnn.yaml

sbatch scripts/run_train.sh conf/experiments/panel_a.yaml

# --- sFNC only (orange boxes)
sbatch scripts/run_train.sh conf/experiments/baseline_blend.yaml data.modalities=[sFNC]

sbatch scripts/run_train.sh conf/experiments/meanmlp.yaml         data.modalities=[sFNC]

sbatch scripts/run_train.sh conf/experiments/brainnetcnn.yaml     data.modalities=[sFNC]

sbatch scripts/run_train.sh conf/experiments/panel_a.yaml         data.modalities=[sFNC]




The four architectures:

baseline_blend.yaml — late fusion. Each modality gets its own small MLP producing a 64-d embedding; the four embeddings are averaged; a classifier reads the average. Modality identity is gone by the time anything is classified.

meanmlp.yaml — early fusion. All modalities concatenated into one long vector first, then a single MLP over the whole thing. This is cvbench’s winning architecture. Same ingredients as the baseline, opposite order — and that’s the point of having both.

brainnetcnn.yaml — connectivity-shaped convolutions. Rebuilds sFNC’s 1,378 edges into the 53×53 matrix and applies filters that pool row i and column j for each node pair, which is the only kind of convolution that means anything on an adjacency matrix. GM/CSF/fALFF have no such structure, so they keep ordinary MLP encoders.

panel_a.yaml — the transparent layer. Everything projects onto one shared graph of 53 named Neuromark nodes, and the classifier can see nothing except that graph’s output. The only arm you can read a result out of.

The config default is: [GM, CSF, FALFF, sFNC]

Adding data.modalities=[sFNC] is sFNC connectivity only.

