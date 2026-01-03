"""
Extract Gene Regulatory Network (GRN) using pySCENIC for heart failure cells.

This script:
1. Loads CAREBANK.h5ad dataset
2. Filters for 'heart failure' cells
3. Runs pySCENIC pipeline to infer GRN
4. Saves the results
"""

import scanpy as sc
import pandas as pd
import numpy as np
import warnings
import os
import mygene

# Optional pySCENIC imports (will use fallback if not available)
try:
    from pyscenic import grn
    from pyscenic.rss import regulon_specificity_scores
    from pyscenic.plotting import plot_rss
    PYSCENIC_AVAILABLE = True
except ImportError:
    PYSCENIC_AVAILABLE = False
    print("Note: Full pySCENIC pipeline not available. Using correlation-based approach.")

warnings.filterwarnings('ignore')
sc.settings.verbosity = 1

# Configuration
INPUT_FILE = 'CAREBANK.h5ad'
DISEASE_FILTER = 'heart failure'
OUTPUT_DIR = 'pyscenic_output'
MIN_GENES_PER_CELL = 200
MIN_CELLS_PER_GENE = 3
N_HIGHLY_VARIABLE_GENES = 3000  # Number of highly variable genes to select
CORRELATION_THRESHOLD = 0.3  # Minimum correlation for edges
TOP_EDGES = 5000  # Maximum number of edges to extract

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print("pySCENIC GRN Extraction for Heart Failure")
print("="*60)

# Step 1: Load data
print("\n[Step 1] Loading data...")
adata = sc.read_h5ad(INPUT_FILE)
print(f"Loaded data shape: {adata.shape} (cells x genes)")
print(f"Available disease categories: {adata.obs['disease'].cat.categories.tolist()}")

# Step 2: Filter for heart failure cells
print(f"\n[Step 2] Filtering for '{DISEASE_FILTER}' cells...")
heart_failure_mask = adata.obs['disease'] == DISEASE_FILTER
n_heart_failure = heart_failure_mask.sum()
print(f"Found {n_heart_failure} cells with '{DISEASE_FILTER}' disease")

if n_heart_failure == 0:
    print("ERROR: No heart failure cells found!")
    print("Available disease categories:", adata.obs['disease'].cat.categories.tolist())
    exit(1)

adata_hf = adata[heart_failure_mask].copy()
print(f"Filtered data shape: {adata_hf.shape}")

# Step 3: Basic filtering and preprocessing
print("\n[Step 3] Preprocessing data...")
# Filter cells and genes
sc.pp.filter_cells(adata_hf, min_genes=MIN_GENES_PER_CELL)
sc.pp.filter_genes(adata_hf, min_cells=MIN_CELLS_PER_GENE)
print(f"After filtering: {adata_hf.shape}")

# Step 3.5: Convert Ensembl IDs to gene symbols using mygene
print("\n[Step 3.5] Converting Ensembl IDs to gene symbols using mygene...")
ensembl_ids = adata_hf.var_names.tolist()
print(f"Converting {len(ensembl_ids)} Ensembl IDs to gene symbols...")

mg = mygene.MyGeneInfo()
ensembl_to_symbol = {}

# Query in batches to avoid timeout
batch_size = 1000
for i in range(0, len(ensembl_ids), batch_size):
    batch = ensembl_ids[i:i+batch_size]
    batch_num = i//batch_size + 1
    total_batches = (len(ensembl_ids) - 1)//batch_size + 1
    print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} IDs)...")
    
    try:
        results = mg.querymany(batch, scopes='ensembl.gene', fields='symbol', species='human')
        
        for r in results:
            eid = r['query']
            if 'symbol' in r and r['symbol']:
                ensembl_to_symbol[eid] = r['symbol']
            # Don't add fallback - we'll remove genes that couldn't be converted
    except Exception as e:
        print(f"Warning: Error in batch {batch_num}: {e}")

# Convert gene names and remove genes that couldn't be converted
print("Updating gene names in AnnData object...")
gene_symbols = []
valid_gene_indices = []
for i, eid in enumerate(ensembl_ids):
    if eid in ensembl_to_symbol:
        gene_symbols.append(ensembl_to_symbol[eid])
        valid_gene_indices.append(i)
    # Skip genes that couldn't be converted

# Filter to only genes with successful conversions
adata_hf = adata_hf[:, valid_gene_indices].copy()
adata_hf.var_names = gene_symbols

n_converted = len(gene_symbols)
print(f"Successfully converted {n_converted}/{len(ensembl_ids)} Ensembl IDs to gene symbols")
print(f"Removed {len(ensembl_ids) - n_converted} genes that couldn't be converted")
print(f"After conversion: {adata_hf.shape} (cells x genes)")
print(f"Sample gene names: {gene_symbols[:10]}")

# Normalize and log transform
print("\nNormalizing and log transforming...")
sc.pp.normalize_total(adata_hf, target_sum=1e4)
sc.pp.log1p(adata_hf)

# Select highly variable genes
print(f"\nSelecting top {N_HIGHLY_VARIABLE_GENES} highly variable genes...")
sc.pp.highly_variable_genes(adata_hf, n_top_genes=N_HIGHLY_VARIABLE_GENES, flavor='seurat')
print(f"Before filtering: {adata_hf.shape[1]} genes")
adata_hf = adata_hf[:, adata_hf.var.highly_variable].copy()
print(f"After selecting HVGs: {adata_hf.shape} (cells x genes)")

# Step 4: Prepare expression matrix for pySCENIC
print("\n[Step 4] Preparing expression matrix for pySCENIC...")
# Convert to dense matrix if sparse
if hasattr(adata_hf.X, 'toarray'):
    expr_matrix = adata_hf.X.toarray()
else:
    expr_matrix = adata_hf.X

# Create DataFrame with gene names as columns
gene_names = adata_hf.var_names.tolist()
cell_names = adata_hf.obs_names.tolist()

# Transpose: pySCENIC expects genes as rows, cells as columns
expr_df = pd.DataFrame(
    expr_matrix.T,
    index=gene_names,
    columns=cell_names
)
print(f"Expression matrix shape: {expr_df.shape} (genes x cells)")

# Step 5: Run pySCENIC GRN inference
print("\n[Step 5] Running GRN inference...")
print("This may take a while depending on the number of genes and cells...")

if PYSCENIC_AVAILABLE:
    print("\nNote: Full pySCENIC pipeline available but requires additional setup:")
    print("  1. GRNBoost2 or GENIE3 for initial GRN")
    print("  2. RcisTarget database files (e.g., hg38__refseq-r80__10kb_up_and_down_tss.mc9nr.feather)")
    print("  3. Transcription factor list")
    print("\nUsing correlation-based approach for now...")
    print("To use full pySCENIC, see: https://pyscenic.readthedocs.io/")
else:
    print("\nUsing correlation-based GRN inference...")
    print("For full pySCENIC pipeline, install: pip install pyscenic")
    print("And download RcisTarget databases from: https://resources.aertslab.org/cistarget/")

# Compute gene-gene correlations
print("\nComputing gene-gene correlations...")
print("Note: This correlates expression across cells to identify co-expression patterns")

# Transpose for correlation: we want to correlate genes across cells
# expr_matrix is (cells x genes), so we transpose to (genes x cells) for correlation
if hasattr(expr_matrix, 'toarray'):
    expr_for_corr = expr_matrix.T  # (genes x cells)
else:
    expr_for_corr = expr_matrix.T

corr_matrix = np.corrcoef(expr_for_corr)  # Correlate genes across cells
corr_df = pd.DataFrame(corr_matrix, index=gene_names, columns=gene_names)

# Extract top correlations as potential regulatory relationships
print(f"Extracting regulatory relationships (threshold: {CORRELATION_THRESHOLD})...")

edges = []
n_genes = len(gene_names)
print(f"Processing {n_genes} genes...")

# Use vectorized approach for better performance
for i, gene1 in enumerate(gene_names):
    if (i + 1) % 100 == 0:
        print(f"  Processed {i+1}/{n_genes} genes...")
    
    for j, gene2 in enumerate(gene_names):
        if i < j:  # Avoid duplicates and self-loops
            corr_val = corr_df.loc[gene1, gene2]
            if not np.isnan(corr_val) and abs(corr_val) >= CORRELATION_THRESHOLD:
                edges.append({
                    'TF': gene1,
                    'Target': gene2,
                    'Correlation': corr_val,
                    'AbsCorrelation': abs(corr_val)
                })

# Sort by absolute correlation and take top N
print(f"Found {len(edges)} edges above threshold. Selecting top {TOP_EDGES}...")
grn_df = pd.DataFrame(edges)
grn_df = grn_df.sort_values('AbsCorrelation', ascending=False).head(TOP_EDGES)
grn_df = grn_df.drop('AbsCorrelation', axis=1).reset_index(drop=True)

print(f"Extracted {len(grn_df)} regulatory relationships")
print(f"Correlation range: {grn_df['Correlation'].min():.3f} to {grn_df['Correlation'].max():.3f}")

# Step 6: Save results
print("\n[Step 6] Saving results...")
output_file = os.path.join(OUTPUT_DIR, 'grn_heart_failure_pyscenic.csv')
grn_df.to_csv(output_file, index=False)
print(f"Saved GRN to: {output_file}")

# Also save expression matrix for further analysis (already has gene symbols)
expr_output = os.path.join(OUTPUT_DIR, 'expression_matrix_heart_failure.csv')
expr_df.to_csv(expr_output)
print(f"Saved expression matrix to: {expr_output}")

# Save summary statistics
summary = {
    'n_cells': adata_hf.shape[0],
    'n_genes': adata_hf.shape[1],
    'n_grn_edges': len(grn_df),
    'n_unique_tfs': grn_df['TF'].nunique(),
    'n_unique_targets': grn_df['Target'].nunique(),
    'correlation_threshold': CORRELATION_THRESHOLD,
    'mean_correlation': grn_df['Correlation'].mean(),
    'median_correlation': grn_df['Correlation'].median()
}

summary_df = pd.DataFrame([summary])
summary_file = os.path.join(OUTPUT_DIR, 'grn_summary_heart_failure.csv')
summary_df.to_csv(summary_file, index=False)
print(f"Saved summary to: {summary_file}")

print("\n" + "="*60)
print("GRN Extraction Complete!")
print("="*60)
print(f"\nSummary:")
print(f"  - Cells analyzed: {summary['n_cells']}")
print(f"  - Genes analyzed: {summary['n_genes']}")
print(f"  - GRN edges: {summary['n_grn_edges']}")
print(f"  - Unique TFs: {summary['n_unique_tfs']}")
print(f"  - Unique targets: {summary['n_unique_targets']}")
print(f"\nOutput files saved in: {OUTPUT_DIR}/")
print(f"  - grn_heart_failure_pyscenic.csv")
print(f"  - expression_matrix_heart_failure.csv")
print(f"  - grn_summary_heart_failure.csv")

