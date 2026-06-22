import os
import pandas as pd
# pyrefly: ignore [missing-import]
from settings import DATA_ROOT

def _load_base_data():
    """Helper to cleanly load the raw circRNA counts and clinical metadata."""
    counts_df = pd.read_csv(os.path.join(DATA_ROOT, 'circRNA_counts.csv'))
    
    metadata_path = os.path.join(DATA_ROOT, 'clean_clinical_metadata.csv')
    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
    else:
        metadata_df = pd.DataFrame()
        
    return counts_df, metadata_df

def load_raw_filtered_circRNAs(criterion='annotated_only', apply_filter=True):
    """
    Filters circRNAs based on raw annotation and junction heuristics.
    
    Parameters:
    - criterion (str): Must be 'annotated_only', 'exon_exon_only', 'junction_type_1', or 'multi_circ_genes_only'.
    - apply_filter (bool): If True, counts_df is filtered. If False, the full dataset is returned.
    
    Returns:
    - (counts_df, metadata_df, target_circRNA_ids)
    """
    counts_df, metadata_df = _load_base_data()
    
    if criterion == 'annotated_only':
        valid_df = counts_df[counts_df['Gene'] != 'not_annotated']
        
    elif criterion == 'exon_exon_only':
        # Strictly filters for circles made entirely of exons
        valid_df = counts_df[counts_df['Start-End Region'] == 'exon-exon']
        
    elif criterion == 'junction_type_1':
        # Filters for canonical splicing signals, even if they occur in intergenic space
        valid_df = counts_df[counts_df['JunctionType'] == 1]
        
    elif criterion == 'multi_circ_genes_only':
        # Exclude 'not_annotated' so they don't incorrectly get grouped as a single gene
        annotated = counts_df[counts_df['Gene'] != 'not_annotated']
        gene_counts = annotated['Gene'].value_counts()
        multi_genes = gene_counts[gene_counts > 1].index
        valid_df = annotated[annotated['Gene'].isin(multi_genes)]
        
    else:
        raise ValueError("criterion must be 'annotated_only', 'exon_exon_only', 'junction_type_1', or 'multi_circ_genes_only'")
        
    # Get exact list of valid targets
    target_ids = valid_df['circRNA_id'].tolist()
    
    if apply_filter:
        counts_df = valid_df.copy()
        
    return counts_df, metadata_df, target_ids


def load_de_filtered_circRNAs(analyses=None, apply_filter=True):
    """
    Filters circRNAs strictly to those originating from significantly differentially expressed genes.
    
    Parameters:
    - analyses (list of int): Optional list of analysis_codes to include (e.g., [1, 2, 5]). 
                              If None, includes all DE genes across all analyses.
    - apply_filter (bool): If True, counts_df is filtered. If False, the full dataset is returned.
    
    Returns:
    - (counts_df, metadata_df, target_circRNA_ids)
    """
    counts_df, metadata_df = _load_base_data()
    
    de_df = pd.read_csv(os.path.join(DATA_ROOT, 'significant_DE_genes.csv'))
    
    if analyses is not None:
        de_df = de_df[de_df['analysis_code'].isin(analyses)]
        
    # Extract unique circRNA_IDs
    target_ids = de_df['circRNA_ID'].dropna().unique().tolist()
    
    if apply_filter:
        counts_df = counts_df[counts_df['circRNA_id'].isin(target_ids)].copy()
        
    return counts_df, metadata_df, target_ids


def load_gsea_filtered_circRNAs(analyses=None, apply_filter=True):
    """
    Filters circRNAs to those driving significantly enriched biological pathways.
    
    Parameters:
    - analyses (list of int): Optional list of analysis_codes to include.
    - apply_filter (bool): If True, counts_df is filtered. If False, the full dataset is returned.
    
    Returns:
    - (counts_df, metadata_df, target_circRNA_ids)
    """
    counts_df, metadata_df = _load_base_data()
    
    gsea_df = pd.read_csv(os.path.join(DATA_ROOT, 'significant_GSEA_genes.csv'))
    
    if analyses is not None:
        gsea_df = gsea_df[gsea_df['analysis_code'].isin(analyses)]
        
    # Deduplicate circRNA_IDs since GSEA pathways overlap heavily
    target_ids = gsea_df['circRNA_ID'].dropna().unique().tolist()
    
    if apply_filter:
        counts_df = counts_df[counts_df['circRNA_id'].isin(target_ids)].copy()
        
    return counts_df, metadata_df, target_ids
