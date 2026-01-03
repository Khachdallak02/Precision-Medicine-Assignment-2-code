"""
Extract Gene Regulatory Network (GRN) using scPRINT for heart failure cells.

This script:
1. Loads CAREBANK.h5ad dataset
2. Filters for 'heart failure' cells
3. Adds organism ontology term
4. Runs scPRINT to infer GRN
5. Saves the results
"""

import warnings
warnings.filterwarnings('ignore')

# IMPORTANT: Connect to lamindb BEFORE importing scPRINT
# scPRINT dependencies check lamindb connection at import time
import lamindb as ln
print("\n[Setup] Connecting to lamindb...")
try:
    ln.connect()
    print("✓ Connected to lamindb instance")
except Exception:
    print("⚠ No existing lamindb instance found. Initializing new instance...")
    ln.setup.init(storage="./lamindb_storage")
    print("✓ Initialized new lamindb instance")

# Now safe to import scPRINT
import scanpy as sc
from scprint import scPrint
from scprint.tasks import GNInfer

sc.settings.verbosity = 1

# Configuration
INPUT_FILE = 'data/main/CAREBANK.h5ad'
DISEASE_FILTER = 'heart failure'
CKPT_PATH = 'small-v1.ckpt'
OUTPUT_FILE = 'heart_failure_grn_scprint.h5ad'

print("="*60)
print("scPRINT GRN Extraction for Heart Failure")
print("="*60)

# Step 1: Load data
print("\n[Step 1] Loading data...")
adata = sc.read_h5ad(INPUT_FILE)
print(f"Loaded data shape: {adata.shape} (cells x genes)")

# Step 2: Filter for heart failure cells
print(f"\n[Step 2] Filtering for '{DISEASE_FILTER}' cells...")
heart_failure_mask = adata.obs['disease'] == DISEASE_FILTER
adata_hf = adata[heart_failure_mask].copy()
print(f"Filtered data shape: {adata_hf.shape}")

# Step 3: Add organism ontology term
print("\n[Step 3] Adding organism ontology term...")
adata_hf.obs['organism_ontology_term_id'] = 'NCBITaxon:9606'

# Step 4: Load scPRINT model
print(f"\n[Step 4] Loading scPRINT model from {CKPT_PATH}...")
model = scPrint.load_from_checkpoint(CKPT_PATH)

# Step 5: Infer gene network
print("\n[Step 5] Inferring gene regulatory network...")
grn_inferer = GNInfer()
grn_adata = grn_inferer(model, adata_hf)

# Aggregate attention heads if needed
if grn_adata.varp['GRN'].ndim > 2:
    print("Aggregating attention heads...")
    grn_adata.varp['GRN'] = grn_adata.varp['GRN'].mean(-1)

# Step 6: Save results
print(f"\n[Step 6] Saving results to {OUTPUT_FILE}...")
grn_adata.write_h5ad(OUTPUT_FILE)

print("\n" + "="*60)
print("GRN Extraction Complete!")
print("="*60)
print(f"\nOutput saved to: {OUTPUT_FILE}")
print(f"GRN shape: {grn_adata.varp['GRN'].shape}")

