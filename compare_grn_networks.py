"""
Compare Gene Regulatory Networks (GRN) from pySCENIC and scPRINT.

This script:
1. Loads GRN networks from pySCENIC (CSV) and scPRINT (H5AD)
2. Compares network statistics and topology metrics
3. Creates visualizations comparing the networks
4. Analyzes overlap, hub genes, and network structure
"""

import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import warnings
from collections import Counter

warnings.filterwarnings('ignore')

# Configuration
DEFAULT_PYSCENIC_FILE = 'pyscenic_output/grn_heart_failure_pyscenic.csv'
DEFAULT_SCPRINT_FILE = 'heart_failure_grn_scprint.h5ad'
OUTPUT_DIR = 'network_comparison'
MIN_CORRELATION = 0.0001  # Minimum correlation threshold
MAX_EDGES = 50000  # Maximum number of edges to keep (for performance)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_grn_from_h5ad(grn_file):
    """Load GRN data from H5AD file (GRnnData format)"""
    try:
        from grnndata import GRNAnnData
        import anndata as ad
    except ImportError as e:
        print("ERROR: GRnnData package not installed.")
        print("Install it with: pip install grnndata")
        print(f"Import error: {e}")
        return None
    
    # Try to import read_h5ad if available
    try:
        from grnndata import read_h5ad
        has_read_h5ad = True
    except (ImportError, AttributeError):
        has_read_h5ad = False
    
    print(f"Loading GRN from H5AD file: {grn_file}")
    try:
        grn_data = None
        
        # Method 1: Try read_h5ad function from grnndata (if available)
        if has_read_h5ad:
            try:
                grn_data = read_h5ad(grn_file)
                print("✓ Loaded using grnndata.read_h5ad()")
            except Exception as e:
                print(f"  Note: read_h5ad() failed: {e}")
                grn_data = None
        
        # Method 2: Load with anndata and wrap in GRNAnnData (fallback)
        if grn_data is None:
            try:
                adata = ad.read_h5ad(grn_file)
                grn_data = GRNAnnData(adata)
                print("✓ Loaded using anndata.read_h5ad() and wrapped in GRNAnnData")
            except Exception as e2:
                print(f"ERROR: Failed to load H5AD file: {e2}")
                raise ValueError(f"Could not load H5AD file. Error: {e2}")
        
        if grn_data is None:
            print("ERROR: Failed to load GRNAnnData object")
            return None
        
        print(f"✓ Successfully loaded GRNAnnData object")
        
        # Get adjacency matrix from GRN
        adj_matrix = None
        
        if hasattr(grn_data, 'varp') and 'GRN' in grn_data.varp:
            adj_matrix = grn_data.varp['GRN']
            print("✓ Found GRN in varp['GRN']")
        elif hasattr(grn_data, 'grn') and grn_data.grn is not None:
            adj_matrix = grn_data.grn
            print("✓ Found GRN in grn attribute")
        elif hasattr(grn_data, 'varp'):
            varp_keys = list(grn_data.varp.keys())
            print(f"Available varp keys: {varp_keys}")
            for key in varp_keys:
                val = grn_data.varp[key]
                if isinstance(val, np.ndarray) or hasattr(val, 'toarray') or hasattr(val, 'shape'):
                    adj_matrix = val
                    print(f"✓ Found GRN in varp['{key}']")
                    break
        
        if adj_matrix is None:
            print("ERROR: Could not find GRN matrix in H5AD file")
            return None
        
        # Handle sparse matrices
        if hasattr(adj_matrix, 'toarray'):
            print("  Converting sparse matrix to dense...")
            try:
                sparse_shape = adj_matrix.shape
                print(f"  Sparse matrix shape: {sparse_shape}")
                adj_matrix = adj_matrix.toarray()
                print("  ✓ Converted sparse matrix to dense")
            except Exception as e:
                print(f"  ERROR: Failed to convert sparse matrix: {e}")
                return None
        
        # Ensure it's a numpy array
        if not isinstance(adj_matrix, np.ndarray):
            adj_matrix = np.array(adj_matrix)
        
        print(f"✓ GRN matrix shape: {adj_matrix.shape}")
        
        # Get gene names
        gene_names = grn_data.var_names.tolist()
        print(f"✓ Found {len(gene_names)} genes")
        
        # Convert adjacency matrix to edge list
        print("Converting adjacency matrix to edge list...")
        edges = []
        n_genes = len(gene_names)
        
        if adj_matrix.shape[0] != n_genes or adj_matrix.shape[1] != n_genes:
            print(f"WARNING: Matrix shape {adj_matrix.shape} doesn't match gene count {n_genes}")
            n_genes = min(adj_matrix.shape[0], adj_matrix.shape[1], n_genes)
        
        for i in range(n_genes):
            for j in range(n_genes):
                weight = adj_matrix[i, j]
                if weight != 0 and not np.isnan(weight) and not np.isinf(weight):
                    edges.append({
                        'TF': gene_names[i],
                        'Target': gene_names[j],
                        'Correlation': float(weight)
                    })
        
        grn_df = pd.DataFrame(edges)
        print(f"✓ Extracted {len(grn_df)} edges from adjacency matrix")
        
        return grn_df
        
    except Exception as e:
        print(f"ERROR: Failed to load H5AD file: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_grn_data(grn_file, network_name="Network", min_correlation=MIN_CORRELATION, max_edges=None):
    """Load GRN data from CSV or H5AD file"""
    if not os.path.exists(grn_file):
        print(f"ERROR: GRN file not found: {grn_file}")
        return None
    
    print(f"\n{'='*60}")
    print(f"Loading {network_name}")
    print(f"{'='*60}")
    
    file_ext = os.path.splitext(grn_file)[1].lower()
    
    if file_ext == '.h5ad':
        grn_df = load_grn_from_h5ad(grn_file)
    elif file_ext == '.csv':
        print(f"Detected CSV file format")
        grn_df = pd.read_csv(grn_file)
        print(f"Loaded {len(grn_df)} regulatory relationships")
    else:
        print(f"ERROR: Unsupported file format: {file_ext}")
        return None
    
    if grn_df is None:
        return None
    
    # Validate required columns
    required_cols = ['TF', 'Target']
    if not all(col in grn_df.columns for col in required_cols):
        print(f"ERROR: File must contain columns: {required_cols}")
        return None
    
    # Check for Correlation column
    if 'Correlation' not in grn_df.columns:
        if 'Weight' in grn_df.columns:
            grn_df['Correlation'] = grn_df['Weight']
        elif 'weight' in grn_df.columns:
            grn_df['Correlation'] = grn_df['weight']
        else:
            grn_df['Correlation'] = 1.0
    
    # Filter out edges with correlations below threshold
    initial_count = len(grn_df)
    grn_df = grn_df[grn_df['Correlation'].abs() >= min_correlation].copy()
    filtered_count = len(grn_df)
    
    if removed_count := initial_count - filtered_count:
        print(f"Filtered out {removed_count} edges with |correlation| < {min_correlation}")
    
    # If still too many edges, select top N by absolute correlation
    if max_edges and len(grn_df) > max_edges:
        print(f"Selecting top {max_edges} edges by absolute correlation...")
        grn_df['AbsCorrelation'] = grn_df['Correlation'].abs()
        grn_df = grn_df.sort_values('AbsCorrelation', ascending=False).head(max_edges)
        grn_df = grn_df.drop('AbsCorrelation', axis=1)
        print(f"Selected top {len(grn_df)} edges")
    
    print(f"Final edge count: {len(grn_df)}")
    return grn_df


def create_networkx_graph(grn_df, network_name="Network"):
    """Create a NetworkX graph from GRN DataFrame (optimized)"""
    print(f"\nCreating NetworkX graph for {network_name}...")
    G = nx.DiGraph()
    
    # Use vectorized operations for faster graph creation
    edges = [(str(tf), str(target), {'correlation': corr, 'weight': abs(corr)}) 
             for tf, target, corr in zip(grn_df['TF'], grn_df['Target'], grn_df['Correlation'])]
    
    G.add_edges_from(edges)
    
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    
    return G


def calculate_network_statistics(G, network_name):
    """Calculate essential network statistics (optimized for speed)"""
    stats = {}
    
    # Basic statistics
    stats['n_nodes'] = G.number_of_nodes()
    stats['n_edges'] = G.number_of_edges()
    stats['density'] = nx.density(G)
    
    # Degree statistics (fast)
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    total_degrees = dict(G.degree())
    
    in_deg_vals = list(in_degrees.values())
    out_deg_vals = list(out_degrees.values())
    total_deg_vals = list(total_degrees.values())
    
    stats['avg_in_degree'] = np.mean(in_deg_vals) if in_deg_vals else 0
    stats['avg_out_degree'] = np.mean(out_deg_vals) if out_deg_vals else 0
    stats['avg_total_degree'] = np.mean(total_deg_vals) if total_deg_vals else 0
    stats['max_in_degree'] = max(in_deg_vals) if in_deg_vals else 0
    stats['max_out_degree'] = max(out_deg_vals) if out_deg_vals else 0
    stats['max_total_degree'] = max(total_deg_vals) if total_deg_vals else 0
    
    # Identify TFs and targets (fast)
    tfs = {node for node in G.nodes() if out_degrees[node] > 0}
    targets = {node for node in G.nodes() if in_degrees[node] > 0}
    
    stats['n_tfs'] = len(tfs)
    stats['n_targets'] = len(targets)
    stats['n_both'] = len(tfs & targets)
    
    # Hub identification (top 10 by total degree) - fast
    sorted_nodes = sorted(total_degrees.items(), key=lambda x: x[1], reverse=True)
    stats['top_hubs'] = [node for node, _ in sorted_nodes[:10]]
    
    # Centrality measures (only for smaller networks or skip if too large)
    n_nodes = stats['n_nodes']
    if n_nodes < 5000:  # Only calculate for smaller networks
        try:
            print(f"  Calculating betweenness centrality for {network_name}...")
            betweenness = nx.betweenness_centrality(G)
            stats['avg_betweenness'] = np.mean(list(betweenness.values()))
            stats['max_betweenness'] = max(betweenness.values()) if betweenness else 0
        except:
            stats['avg_betweenness'] = 0
            stats['max_betweenness'] = 0
        
        try:
            print(f"  Calculating PageRank for {network_name}...")
            pagerank = nx.pagerank(G)
            stats['avg_pagerank'] = np.mean(list(pagerank.values()))
            stats['max_pagerank'] = max(pagerank.values()) if pagerank else 0
        except:
            stats['avg_pagerank'] = 0
            stats['max_pagerank'] = 0
    else:
        print(f"  Skipping centrality calculations for {network_name} (network too large: {n_nodes} nodes)")
        stats['avg_betweenness'] = 0
        stats['max_betweenness'] = 0
        stats['avg_pagerank'] = 0
        stats['max_pagerank'] = 0
    
    # Clustering (only for smaller networks)
    if n_nodes < 5000:
        try:
            print(f"  Calculating clustering coefficient for {network_name}...")
            G_undirected = G.to_undirected()
            clustering = nx.clustering(G_undirected)
            stats['avg_clustering'] = np.mean(list(clustering.values()))
        except:
            stats['avg_clustering'] = 0
    else:
        stats['avg_clustering'] = 0
    
    return stats, in_degrees, out_degrees, total_degrees, tfs, targets


def compare_networks(G1, G2, name1="Network 1", name2="Network 2"):
    """Compare two networks and calculate overlap metrics"""
    print(f"\n{'='*60}")
    print(f"Comparing {name1} vs {name2}")
    print(f"{'='*60}")
    
    nodes1 = set(G1.nodes())
    nodes2 = set(G2.nodes())
    
    edges1 = set(G1.edges())
    edges2 = set(G2.edges())
    
    # Node overlap
    common_nodes = nodes1 & nodes2
    unique_nodes1 = nodes1 - nodes2
    unique_nodes2 = nodes2 - nodes1
    
    # Edge overlap
    common_edges = edges1 & edges2
    unique_edges1 = edges1 - edges2
    unique_edges2 = edges2 - edges1
    
    # Jaccard similarity
    node_jaccard = len(common_nodes) / len(nodes1 | nodes2) if (nodes1 | nodes2) else 0
    edge_jaccard = len(common_edges) / len(edges1 | edges2) if (edges1 | edges2) else 0
    
    comparison = {
        'nodes1': len(nodes1),
        'nodes2': len(nodes2),
        'common_nodes': len(common_nodes),
        'unique_nodes1': len(unique_nodes1),
        'unique_nodes2': len(unique_nodes2),
        'node_jaccard': node_jaccard,
        'edges1': len(edges1),
        'edges2': len(edges2),
        'common_edges': len(common_edges),
        'unique_edges1': len(unique_edges1),
        'unique_edges2': len(unique_edges2),
        'edge_jaccard': edge_jaccard,
        'common_nodes_set': common_nodes,
        'common_edges_set': common_edges
    }
    
    print(f"\nNode Comparison:")
    print(f"  {name1}: {len(nodes1)} nodes")
    print(f"  {name2}: {len(nodes2)} nodes")
    print(f"  Common: {len(common_nodes)} nodes ({node_jaccard*100:.2f}% Jaccard similarity)")
    print(f"  Unique to {name1}: {len(unique_nodes1)} nodes")
    print(f"  Unique to {name2}: {len(unique_nodes2)} nodes")
    
    print(f"\nEdge Comparison:")
    print(f"  {name1}: {len(edges1)} edges")
    print(f"  {name2}: {len(edges2)} edges")
    print(f"  Common: {len(common_edges)} edges ({edge_jaccard*100:.2f}% Jaccard similarity)")
    print(f"  Unique to {name1}: {len(unique_edges1)} edges")
    print(f"  Unique to {name2}: {len(unique_edges2)} edges")
    
    return comparison


def plot_average_clustering(stats1, stats2, name1, name2):
    """Plot Average Clustering Coefficient comparison"""
    print("Creating Average Clustering Coefficient plot...")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x = np.arange(2)
    width = 0.6
    
    values = [stats1['avg_clustering'], stats2['avg_clustering']]
    labels = [name1, name2]
    colors = ['#FF6B6B', '#4ECDC4']
    
    bars = ax.bar(x, values, width, color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_ylabel('Average Clustering Coefficient', fontsize=12)
    ax.set_title('Average Clustering Coefficient Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    output_file = os.path.join(OUTPUT_DIR, 'average_clustering_coefficient.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot_average_pagerank(stats1, stats2, name1, name2):
    """Plot Average PageRank comparison"""
    print("Creating Average PageRank plot...")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x = np.arange(2)
    width = 0.6
    
    values = [stats1['avg_pagerank'], stats2['avg_pagerank']]
    labels = [name1, name2]
    colors = ['#FF6B6B', '#4ECDC4']
    
    bars = ax.bar(x, values, width, color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_ylabel('Average PageRank', fontsize=12)
    ax.set_title('Average PageRank Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.6f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    output_file = os.path.join(OUTPUT_DIR, 'average_pagerank.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot_in_degree_distribution(in_deg1, in_deg2, name1, name2):
    """Plot In-Degree distribution with logged frequency axis"""
    print("Creating In-Degree distribution plot (log frequency)...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    in_deg_vals1 = list(in_deg1.values())
    in_deg_vals2 = list(in_deg2.values())
    
    # Use adaptive binning
    bins = min(50, max(20, int(np.sqrt(max(len(in_deg_vals1), len(in_deg_vals2))))))
    
    counts1, bins1, patches1 = ax.hist(in_deg_vals1, bins=bins, alpha=0.6, 
                                       label=name1, color='#FF6B6B', edgecolor='black')
    counts2, bins2, patches2 = ax.hist(in_deg_vals2, bins=bins, alpha=0.6, 
                                       label=name2, color='#4ECDC4', edgecolor='black')
    
    # Set y-axis to log scale
    ax.set_yscale('log')
    
    ax.set_xlabel('In-Degree', fontsize=12)
    ax.set_ylabel('Frequency (log scale)', fontsize=12)
    ax.set_title('In-Degree Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    output_file = os.path.join(OUTPUT_DIR, 'in_degree_distribution.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot_out_degree_distribution(out_deg1, out_deg2, name1, name2):
    """Plot Out-Degree distribution with normal frequency axis"""
    print("Creating Out-Degree distribution plot...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    out_deg_vals1 = list(out_deg1.values())
    out_deg_vals2 = list(out_deg2.values())
    
    # Use adaptive binning
    bins = min(50, max(20, int(np.sqrt(max(len(out_deg_vals1), len(out_deg_vals2))))))
    
    ax.hist(out_deg_vals1, bins=bins, alpha=0.6, label=name1, color='#FF6B6B', edgecolor='black')
    ax.hist(out_deg_vals2, bins=bins, alpha=0.6, label=name2, color='#4ECDC4', edgecolor='black')
    
    ax.set_xlabel('Out-Degree', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Out-Degree Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = os.path.join(OUTPUT_DIR, 'out_degree_distribution.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def main():
    """Main function to compare networks"""
    parser = argparse.ArgumentParser(
        description='Compare GRN networks from pySCENIC and scPRINT',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare default files:
  python compare_grn_networks.py
  
  # Compare custom files:
  python compare_grn_networks.py --pyscenic-file custom_pyscenic.csv --scprint-file custom_scprint.h5ad
        """
    )
    parser.add_argument('--pyscenic-file', default=DEFAULT_PYSCENIC_FILE,
                        help=f'Path to pySCENIC GRN CSV file (default: {DEFAULT_PYSCENIC_FILE})')
    parser.add_argument('--scprint-file', default=DEFAULT_SCPRINT_FILE,
                        help=f'Path to scPRINT GRN H5AD file (default: {DEFAULT_SCPRINT_FILE})')
    parser.add_argument('--min-correlation', type=float, default=MIN_CORRELATION,
                        help=f'Minimum correlation threshold (default: {MIN_CORRELATION})')
    parser.add_argument('--max-edges', type=int, default=MAX_EDGES,
                        help=f'Maximum number of edges to keep per network (default: {MAX_EDGES})')
    
    args = parser.parse_args()
    
    print("="*60)
    print("GRN Network Comparison: pySCENIC vs scPRINT")
    print("="*60)
    print(f"Max edges per network: {args.max_edges}")
    print("="*60)
    
    # Load networks
    grn_df1 = load_grn_data(args.pyscenic_file, "pySCENIC", args.min_correlation, args.max_edges)
    if grn_df1 is None:
        print("\nERROR: Failed to load pySCENIC network. Exiting.")
        return
    
    grn_df2 = load_grn_data(args.scprint_file, "scPRINT", args.min_correlation, args.max_edges)
    if grn_df2 is None:
        print("\nERROR: Failed to load scPRINT network. Exiting.")
        return
    
    # Create NetworkX graphs
    G1 = create_networkx_graph(grn_df1, "pySCENIC")
    G2 = create_networkx_graph(grn_df2, "scPRINT")
    
    # Calculate statistics
    stats1, in_deg1, out_deg1, total_deg1, tfs1, targets1 = calculate_network_statistics(G1, "pySCENIC")
    stats2, in_deg2, out_deg2, total_deg2, tfs2, targets2 = calculate_network_statistics(G2, "scPRINT")
    
    # Compare networks (for statistics only, no file saving)
    comparison = compare_networks(G1, G2, "pySCENIC", "scPRINT")
    
    # Create the 4 requested plots
    plot_average_clustering(stats1, stats2, "pySCENIC", "scPRINT")
    plot_average_pagerank(stats1, stats2, "pySCENIC", "scPRINT")
    plot_in_degree_distribution(in_deg1, in_deg2, "pySCENIC", "scPRINT")
    plot_out_degree_distribution(out_deg1, out_deg2, "pySCENIC", "scPRINT")
    
    print("\n" + "="*60)
    print("Comparison complete!")
    print("="*60)
    print(f"\nOutput files saved to: {OUTPUT_DIR}/")
    print("  - average_clustering_coefficient.png")
    print("  - average_pagerank.png")
    print("  - in_degree_distribution.png")
    print("  - out_degree_distribution.png")


if __name__ == "__main__":
    main()

