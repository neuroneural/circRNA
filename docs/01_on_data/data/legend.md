# Data Dictionaries

## 1. Clinical Metadata (`clean_clinical_metadata.csv`)
This document maps the raw RNA-seq samples to their respective clinical and demographic features.

### Core Identifiers
* **`Sample_ID`**: Original raw sequencing ID (e.g., `sample_20`).
* **`URSI`**: Unique patient identifier across all visits.
* **`Visit`**: Visit number (1, 2, or 3).
* **`map_id`**: URSI combined with Visit (e.g., `M87114859_V1`). **This ID exactly matches the column headers in the counts tables (`circRNA_counts.csv` and `linRNA_counts.csv`).**

### Demographics & Status
* **`Age`**: Patient age.
* **`Sex`**: Patient sex.
* **`Treatment Status`**: Stage of treatment (`Pre`, `During`, or `Post`).
* **`Response`**: Clinical responder status (`Yes` or `No`).

### Clinical Assessment Scores
* **`SHAPSC`**: Snaith-Hamilton Pleasure Scale. A clinical assessment measuring anhedonia (the inability to experience pleasure).
* **`TEPS`**: Temporal Experience of Pleasure Scale. An assessment measuring trait anticipatory and consummatory pleasure.
* **`IDS-C`**: Inventory of Depressive Symptomatology (Clinician-Rated). A scale used to assess the severity of depressive symptoms.
* **`GAD-7`**: Generalized Anxiety Disorder 7-item scale. A brief clinical measure for assessing generalized anxiety disorder severity.
* **`YMRS`**: Young Mania Rating Scale. A clinical rating scale used to assess the severity of manic symptoms.

## 2. Significant DE Genes
This file aggregates all statistically significant genes (`padj < 0.05`) across the 8 Differential Expression analyses.

### Gene & Analysis Info
* **`gene_name`**: HGNC symbol. **Used to merge with circRNA counts.**
* **`gene_id`**: Ensembl Gene ID.
* **`gene_type`**: Gene classification (e.g., `protein_coding`).
* **`analysis_name`**: DE comparison variant (e.g., `Pre_vs_Post_treatment`).
* **`analysis_code`**: Numeric ID (0-7) for the analysis.

### Statistical Metrics
* **`padj`**: Adjusted p-value.
* **`log2FoldChange`**: The effect size. Shows the magnitude and direction (up/down) of the gene's expression change. Crucial for understanding biological response.
* **`baseMean`**: The average baseline expression level. Essential for reliability—a large fold change on a highly expressed gene (large baseMean) is biologically much more robust than the same change on a barely expressed gene.
