# CircRNA Data Mapping & Statistical Analysis Pipeline

This document serves as a roadmap for analysis of circRNA data. The focus of this pipeline is to use **pure Python (Pandas)** to safely map IDs, run high-level statistical exploration, and compare potential filtering criteria.

## Phase 1: Environment Setup & Data Safety
*Goal: Ensure all original raw and processed data remains strictly read-only.*

- [x] **Create Output Directory**: Create a dedicated directory (e.g., `_investigation/02_processed_data_stats`). All subsequent filtered matrices, mapped CSVs, and statistical reports will be saved here.
CREATED /Users/ppopov1/_circRNA/_investigation/01_on_data/data
- [x] **Python Setup**: Initialize a Python 3.12 environment using conda. Call it circrna.Install pandas, numpy, matplotlib, seaborn, torch, jupyter, statannotations. Create a short MD file describing the steps for reproducibility instructions.

## Phase 2: Data Loading & ID Mapping (The "Matching" Layer)
*Goal: Build robust mapping dictionaries/DataFrames to cross-reference subjects and genes across the various files.*

- [x] **Load Datasets**: Load the following files as Pandas DataFrames:
    - `circ_counts_post_filtering_circ_linear_and_1cnt_in.25samples_ERVIN_202504.csv`
    - `demo_data.csv`
    - `PrevsPost_ERVIN_DE_results_20250612.csv`
    - `PrevsPost_ERVIN_DE_GSEA_results*.csv` (Check column names to verify it already contains only significant pathways).
- [x] **Subject ID Matching**: 
    - Read `RNAseq/demo_data.csv` and extract the metadata for the sample IDs.
- [ ] **Gene ID Matching**:
    - Develop a mapping strategy to cross-reference Entrez IDs (from GSEA), HGNC Gene Symbols (from DE results), and the `Gene` column in the circRNA counts.

## Phase 3: High-Level Statistical Analysis
*Goal: Understand the baseline distribution and biological composition of the circRNA dataset.*

- [x] **CircRNA Frequency Distributions**:
    - Analyze the distribution of counts across samples.
    - Calculate how many circRNAs are present in 1 subject, 2 subjects, 3 subjects, etc., to understand the sparsity of the data.
- [x] **Biological Origin Statistics**:
    - Calculate the proportion of counts originating from unidentified/unannotated regions (`not_annotated` / intergenic) versus known, annotated genes.

## Phase 4: Evaluating Criteria for Future Training (Feature Exploration)
*Goal: Test different approaches for subsetting the matrix to find the best biological/statistical criteria for future modeling.*

- [x] **Criterion 1: The `exon-exon` Approach**:
    - Filter the matrix to include only `exon-exon` back-splices (highly reliable, Junction Type 1). Analyze how many features/counts are retained.
- [ ] **Criterion 2: The Differential Expression (DE) Approach**:
    - Filter the matrix using only features that overlap with significant genes from the DE results (`padj < 0.05`). Analyze how many features/counts are retained.
- [ ] **Criterion 3: Unannotated vs. Known**:
    - Compare the statistics of using only known genes vs. including `not_annotated` circRNA loops.
- [ ] **Criterion 4: The GSEA Pathway Approach**:
    - Filter the matrix using only the `core_enrichment` genes from significant GSEA pathways. Analyze how many features/counts are retained compared to the raw DE approach.
- [ ] **Compare & Discuss**:
    - Compare the overlaps and differences between these filtering criteria.
    - Review the results and discuss with colleagues to determine the optimal subsetting strategy for eventual predictive modeling.
