```bash
conda create -n circrna python=3.12 pandas numpy matplotlib seaborn jupyter -y
conda run -n circrna pip install torch statannotations openpyxl pyfastx ml4fmri
# for the MDD data prep (docs/02): reading ICA-timecourse NIfTIs + postprocess .mat
conda run -n circrna pip install nibabel h5py
conda activate circrna
```
