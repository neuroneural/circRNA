# Clinical Metadata Legend

This document describes the columns present in the `clean_clinical_metadata.csv` file, which maps the raw RNA-seq samples to their respective clinical and demographic features.

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
