# circRNA Data Investigation Notes

## Tissue Source & Sampling
* **Origin**: Samples were extracted from **PAXgene tubes**, which are used to collect and stabilize RNA from **whole blood**.
* **Biological Implications**: Because mature red blood cells lack nuclei, this RNA primarily represents the transcriptome of **white blood cells** (leukocytes like neutrophils, lymphocytes, and monocytes).
* **Biomarker Context**: Models trained on this data will be detecting **systemic, blood-based biomarkers** of psychiatric conditions, rather than direct brain-tissue signatures.

## General Analysis pipeline
* **Raw Data**: FASTQ files (concatenated top-ups where applicable).
* **Linear Processing**: `STAR` -> `Salmon` (Outputs: `quant.sf` for counts, `.tin.xls` for integrity, `.cram` for alignments).
* **Circular Processing**: `STAR` specific chimeric pipeline (Outputs: `.Chimeric.out.junction` for circular back-splicing, high-confidence filtered CSV).

## Columns in CircRNA Counts Table `circ_counts_...csv`

| Column Name | Description |
| :--- | :--- |
| **`Chr`** | Chromosome (e.g., `chr7`) |
| **`Start` / `End`** | Genomic coordinates of the back-splice (Defines the loop boundaries) |
| **`Gene`** | Parent gene the circRNA originates from (`not_annotated` if intergenic) |
| **`JunctionType`** | Type of splice site used (`1` = Standard (annotated), `2` = Novel (unannotated)) |
| **`Strand`** | DNA strand transcribed from (`+` or `-`) |
| **`Start-End Region`** | Structural biology of the loop (`exon-exon`, `intergenic-intergenic`, etc.) |
| **`chr_start_end_strand`** | Unique identifier (**Use this as the Feature ID in ML models**) |
| **`sample_*`** | Count data (The number of times this circRNA was seen in the sample) |

### Junction Type Biology

| Type | Name | Meaning | Reliability for ML |
| :---: | :--- | :--- | :--- |
| **1** | Annotated | Formed using known, standard splice sites (usually exon boundaries). | High (well-studied, biological mechanism understood) |
| **2** | Unannotated | Formed using cryptic or novel splice sites (often in intergenic regions). | Lower (could be noise, splicing errors, or novel biomarkers) |

### Start-End Region Biology

This column describes the genetic "neighborhood" where the two ends of the circle are anchored.

| Region Type | Description | Link to Junction Type |
| :--- | :--- | :--- |
| **`exon-exon`** | The circle is formed purely of known exons (e.g., Exon 3 looped back to Exon 2). | Usually Type 1 |
| **`intron-intron`** | The circle consists entirely of intronic (non-coding) material that escaped degradation. | Usually Type 2 |
| **`intergenic-intergenic`** | The circle formed entirely from a region of the genome without a known gene ("dark matter"). | Almost always Type 2 |
| **`intron-intergenic`** | A hybrid circle where one end is anchored in a known intron, and the other in intergenic space. | Almost always Type 2 |

## Types of Differential Expression (DE) Analyses Performed

The DE analysis was run using the counts as the response variable against **8 different predictors of interest**. This means you have multiple lenses through which to select features for Machine Learning depending on your target variable (`y`):

*   **Categorical / Treatment Status (2 comparisons):**
    1.  Pre-treatment vs. Post-treatment
    2.  Pre-treatment vs. During-treatment
*   **Numeric / Clinical Measures (5 comparisons):** 
    *(Identifying genes correlated with symptom severity)*
    3.  Snaith-Hamilton Pleasure Scale
    4.  Temporal Experience of Pleasure Scale
    5.  Generalized Anxiety Disorder 7-item (GAD-7)
    6.  Inventory of Depression Symptoms - Clinician Rated (IDS-C)
    7.  Young Mania Rating Scale (YMRS)
*   **Clinical Outcome (1 comparison):**
    8.  Responders vs. Non-responders (Post-treatment samples only)

## Columns in Differential Expression (DE) Results `*DE_results*.csv`

| Column Name | Description |
| :--- | :--- |
| **`gene_id`** | Ensembl ID (e.g., `ENSG00000242534.2`) (Most precise identifier.) |
| **`gene_type`** | Biotype of the gene (e.g., `protein_coding`, `lncRNA`, `IG_V_gene`) |
| **`gene_name`** | HGNC symbol (e.g., `SDF4`, `DEFA1B`) (**Map this to the `Gene` column in circRNA counts.** Note: One gene can map to multiple circRNAs.) |
| **`baseMean`** | Average expression level of the gene across *all* samples (Very low values mean the gene is barely expressed.) |
| **`log2FoldChange`** | Effect size (Positive = upregulated in post-treatment, Negative = downregulated.) |
| **`lfcSE`** | Standard error of the log2FoldChange estimate (Measure of uncertainty/variance.) |
| **`stat`** | Wald statistic (Used to calculate the p-value.) |
| **`pvalue`** | Unadjusted p-value of the test (Raw probability the change is due to chance.) |
| **`padj`** | Benjamini-Hochberg adjusted p-value (FDR) (**The most critical column.** Filter significant genes using this (e.g., `< 0.05`) before passing to ML.) |
## Gene Set Enrichment Analysis (GSEA) `*GSEA_results*.csv`

Instead of looking at individual genes, GSEA evaluates whether predefined groups of genes (e.g., biological pathways from KEGG) show statistically significant, concordant differences between your two treatment groups. 

**Why use it for Machine Learning?**
It allows for **Pathway-Based Feature Selection**. Instead of selecting 50 random genes with low p-values, you can select the "core enriched" genes from a highly significant biological pathway. This creates a model that is biologically interpretable and often more robust to noise.

### Columns in GSEA Results

| Column Name | Description |
| :--- | :--- |
| **`ID`** | Pathway identifier (e.g., `hsa03010`, usually a KEGG pathway ID) |
| **`Description`** | Human-readable name of the pathway (e.g., "Motor proteins") |
| **`setSize`** | Number of genes from your dataset that belong to this pathway |
| **`enrichmentScore`** | The raw enrichment score (ES) (Degree to which the gene set is overrepresented at the extremes of your ranked list) |
| **`NES`** | Normalized Enrichment Score (Positive = Upregulated in Post-treatment, Negative = Downregulated) |
| **`p.adjust`** | Adjusted p-value (FDR) (**Filter pathways using this, e.g., `< 0.05`**) |
| **`core_enrichment`** | The specific genes (usually Entrez IDs) that drove the pathway's significance (**Extract these IDs, map them back to gene symbols, and use them as your ML features!**) |

---
*Date:* 2026-04-30
*Author:* Pavel Popov
