"""
Convert pySCENIC GRN CSV to GRnnData format and visualize the network.

This script:
1. Loads the GRN edge list from CSV (TF, Target, Correlation)
2. Converts it to GRnnData format (GRNAnnData object)
3. Saves the GRnnData format
4. Visualizes the network using GRnnData's plotting capabilities
"""

import os
import pandas as pd
import numpy as np
import warnings
import argparse

warnings.filterwarnings('ignore')

# Configuration
DEFAULT_GRN_FILE = 'pyscenic_output/grn_heart_failure_pyscenic.csv'
OUTPUT_DIR = 'grnndata_output'
MIN_CORRELATION = 0.4  # Minimum correlation to include
TOP_EDGES = 1000  # Maximum number of edges to include

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_grn_edge_list(grn_file):
    """Load GRN edge list from CSV file"""
    if not os.path.exists(grn_file):
        print(f"ERROR: GRN file not found: {grn_file}")
        return None
    
    print(f"Loading GRN data from: {grn_file}")
    grn_df = pd.read_csv(grn_file)
    print(f"Loaded {len(grn_df)} regulatory relationships")
    print(f"Columns: {grn_df.columns.tolist()}")
    print(f"\nFirst few edges:")
    print(grn_df.head())
    
    return grn_df


def filter_edges(grn_df, min_correlation=MIN_CORRELATION, top_edges=TOP_EDGES):
    """Filter edges by correlation threshold and select top N"""
    print(f"\nFiltering edges (min correlation: {min_correlation})...")
    grn_filtered = grn_df[grn_df['Correlation'].abs() >= min_correlation].copy()
    print(f"After filtering: {len(grn_filtered)} edges")
    
    # Sort by absolute correlation and take top N
    grn_filtered['AbsCorrelation'] = grn_filtered['Correlation'].abs()
    grn_filtered = grn_filtered.sort_values('AbsCorrelation', ascending=False).head(top_edges)
    grn_filtered = grn_filtered.drop('AbsCorrelation', axis=1).reset_index(drop=True)
    print(f"Selected top {len(grn_filtered)} edges for GRnnData conversion")
    
    return grn_filtered


def create_adjacency_matrix(grn_df):
    """Convert edge list to adjacency matrix"""
    print("\nConverting edge list to adjacency matrix...")
    
    # Get all unique genes (both TFs and targets)
    all_genes = sorted(set(grn_df['TF'].unique()) | set(grn_df['Target'].unique()))
    n_genes = len(all_genes)
    print(f"Total unique genes: {n_genes}")
    
    # Create gene to index mapping
    gene_to_idx = {gene: idx for idx, gene in enumerate(all_genes)}
    
    # Initialize adjacency matrix
    adj_matrix = np.zeros((n_genes, n_genes), dtype=np.float32)
    
    # Fill adjacency matrix with correlation values
    for _, row in grn_df.iterrows():
        tf_idx = gene_to_idx[row['TF']]
        target_idx = gene_to_idx[row['Target']]
        adj_matrix[tf_idx, target_idx] = row['Correlation']
    
    print(f"Adjacency matrix shape: {adj_matrix.shape}")
    print(f"Non-zero entries: {np.count_nonzero(adj_matrix)}")
    
    return adj_matrix, all_genes


def convert_to_grnndata(grn_file, min_correlation=MIN_CORRELATION, top_edges=TOP_EDGES):
    """Convert GRN CSV to GRnnData format"""
    try:
        import anndata as ad
        from grnndata import GRNAnnData
    except ImportError as e:
        print(f"ERROR: Required packages not installed: {e}")
        print("\nPlease install GRnnData:")
        print("  pip install grnndata")
        print("\nAlso ensure anndata is installed:")
        print("  pip install anndata")
        return None
    
    # Load and filter edge list
    grn_df = load_grn_edge_list(grn_file)
    if grn_df is None:
        return None
    
    grn_filtered = filter_edges(grn_df, min_correlation, top_edges)
    
    # Convert to adjacency matrix
    adj_matrix, gene_names = create_adjacency_matrix(grn_filtered)
    
    # Create a dummy expression matrix (GRnnData needs AnnData with expression data)
    # We'll use the adjacency matrix as a placeholder, but GRnnData expects cells x genes
    # For visualization purposes, we can create a minimal expression matrix
    n_cells = 100  # Dummy number of cells
    n_genes = len(gene_names)
    
    # Create a minimal expression matrix (random or zeros)
    # In practice, you might want to load actual expression data
    expr_matrix = np.random.rand(n_cells, n_genes).astype(np.float32)
    
    # Create AnnData object
    print("\nCreating AnnData object...")
    adata = ad.AnnData(
        X=expr_matrix,
        var=pd.DataFrame(index=gene_names),
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)])
    )
    
    # Create GRNAnnData object
    # According to GRnnData API: GRNAnnData(adata, grn=grn_matrix)
    print("Creating GRNAnnData object...")
    try:
        # Try the standard API: GRNAnnData(adata, grn=adj_matrix)
        grn_data = GRNAnnData(adata, grn=adj_matrix)
        print("✓ Created GRNAnnData with GRN matrix")
    except TypeError:
        # Fallback: create without grn parameter and set it manually
        print("Note: Using fallback method to set GRN...")
        grn_data = GRNAnnData(adata)
        # Store as adjacency matrix in varp
        grn_data.varp['GRN'] = adj_matrix
    
    # Store edge list and stats in a format compatible with H5AD
    # Store stats as simple key-value pairs (all numeric or string)
    grn_data.uns['grn_stats'] = {
        'n_edges': int(len(grn_filtered)),
        'n_genes': int(n_genes),
        'n_tfs': int(grn_filtered['TF'].nunique()),
        'n_targets': int(grn_filtered['Target'].nunique()),
        'min_correlation': float(min_correlation),
        'correlation_min': float(grn_filtered['Correlation'].min()),
        'correlation_max': float(grn_filtered['Correlation'].max())
    }
    # Note: Edge list is saved separately as CSV, not in H5AD uns (H5AD has issues with complex nested structures)
    
    print("\nGRnnData object created successfully!")
    print(f"  - Genes: {n_genes}")
    print(f"  - Edges: {len(grn_filtered)}")
    print(f"  - TFs: {grn_filtered['TF'].nunique()}")
    print(f"  - Targets: {grn_filtered['Target'].nunique()}")
    
    return grn_data, grn_filtered


def save_grnndata(grn_data, output_file):
    """Save GRnnData object to file"""
    print(f"\nSaving GRnnData object to: {output_file}")
    try:
        grn_data.write(output_file)
        print("✓ Saved successfully!")
    except Exception as e:
        print(f"Warning: Error saving to H5AD format: {e}")
        print("Attempting alternative save method...")
        # Try saving without problematic uns entries
        try:
            # Remove any problematic uns entries temporarily
            temp_uns = grn_data.uns.copy()
            # Keep only simple types
            simple_uns = {k: v for k, v in temp_uns.items() 
                         if isinstance(v, (str, int, float, bool, list, dict)) 
                         and not isinstance(v, dict) or all(isinstance(v2, (str, int, float, bool)) 
                         for v2 in v.values() if isinstance(v, dict))}
            grn_data.uns.clear()
            grn_data.uns.update(simple_uns)
            grn_data.write(output_file)
            print("✓ Saved successfully with simplified metadata!")
        except Exception as e2:
            print(f"Error: Could not save GRnnData object: {e2}")
            raise


def visualize_grnndata(grn_data, output_dir=OUTPUT_DIR):
    """Visualize GRN using GRnnData's plotting capabilities"""
    print("\n" + "="*60)
    print("Visualizing GRN Network")
    print("="*60)
    
    try:
        # Try to use GRnnData's built-in plotting
        print("\nAttempting to plot using GRnnData...")
        
        # GRnnData might have a plot() method
        if hasattr(grn_data, 'plot'):
            try:
                import matplotlib.pyplot as plt
                fig = grn_data.plot()
                if fig is not None:
                    output_file = os.path.join(output_dir, 'grn_network_grnndata.png')
                    plt.savefig(output_file, dpi=300, bbox_inches='tight')
                    print(f"✓ Saved network plot to: {output_file}")
                    plt.close()
            except Exception as e:
                print(f"Note: GRnnData plot() method encountered an issue: {e}")
                print("Falling back to custom visualization...")
        
        # Alternative: Create custom visualization using the adjacency matrix
        print("\nCreating custom network visualization...")
        create_custom_visualization(grn_data, output_dir)
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        print("Creating fallback visualization...")
        create_custom_visualization(grn_data, output_dir)


def create_custom_visualization(grn_data, output_dir):
    """Create custom network visualization using networkx and matplotlib"""
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        
        # Get adjacency matrix and gene names
        adj_matrix = grn_data.varp['GRN']
        gene_names = grn_data.var_names.tolist()
        
        # Create NetworkX graph
        print("Creating NetworkX graph from adjacency matrix...")
        G = nx.DiGraph()
        
        # Add nodes
        G.add_nodes_from(gene_names)
        
        # Add edges from adjacency matrix
        n_genes = len(gene_names)
        edge_count = 0
        for i in range(n_genes):
            for j in range(n_genes):
                if adj_matrix[i, j] != 0:
                    G.add_edge(gene_names[i], gene_names[j], 
                             weight=adj_matrix[i, j],
                             correlation=adj_matrix[i, j])
                    edge_count += 1
        
        print(f"Graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Identify TFs and targets
        out_degree = dict(G.out_degree())
        in_degree = dict(G.in_degree())
        tfs = {node for node in G.nodes() if out_degree[node] > 0}
        targets_only = {node for node in G.nodes() if in_degree[node] > 0 and out_degree[node] == 0}
        
        # Set node attributes
        for node in G.nodes():
            if node in tfs:
                G.nodes[node]['type'] = 'TF'
            elif node in targets_only:
                G.nodes[node]['type'] = 'Target'
            else:
                G.nodes[node]['type'] = 'Isolated'
        
        # Create visualization
        print("Generating network visualization...")
        plt.figure(figsize=(16, 12))
        
        # Use a layout that works well for directed graphs
        try:
            pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
        except:
            pos = nx.circular_layout(G)
        
        # Draw edges
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=[w*2 for w in weights], 
                              edge_color='gray', arrows=True, arrowsize=10, 
                              arrowstyle='->', connectionstyle='arc3,rad=0.1')
        
        # Draw nodes by type
        # TFs
        tf_nodes = [n for n in G.nodes() if G.nodes[n]['type'] == 'TF']
        if tf_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=tf_nodes, 
                                  node_color='yellow', node_size=500,
                                  node_shape='d', alpha=0.8, label='TF')
        
        # Targets
        target_nodes = [n for n in G.nodes() if G.nodes[n]['type'] == 'Target']
        if target_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=target_nodes,
                                  node_color='lightblue', node_size=200,
                                  node_shape='o', alpha=0.6, label='Target')
        
        # Other nodes
        other_nodes = [n for n in G.nodes() if G.nodes[n]['type'] not in ['TF', 'Target']]
        if other_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=other_nodes,
                                  node_color='lightgray', node_size=100,
                                  node_shape='o', alpha=0.4, label='Other')
        
        # Draw labels for TFs only (to avoid clutter)
        labels = {node: node for node in tf_nodes}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')
        
        plt.title('Gene Regulatory Network (GRnnData Format)\nHeart Failure', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.legend(loc='upper right', fontsize=10)
        plt.axis('off')
        
        # Save figure
        output_file = os.path.join(output_dir, 'grn_network_grnndata.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved network visualization to: {output_file}")
        plt.close()
        
        # Also create a summary statistics plot
        create_statistics_plot(G, tfs, targets_only, output_dir)
        
    except ImportError as e:
        print(f"Error: Required packages not installed: {e}")
        print("Please install: pip install networkx matplotlib")
    except Exception as e:
        print(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()


def create_statistics_plot(G, tfs, targets_only, output_dir):
    """Create statistics plots for the network"""
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
        
        # Calculate statistics
        in_degree = dict(G.in_degree())
        out_degree = dict(G.out_degree())
        degree = dict(G.degree())
        betweenness = nx.betweenness_centrality(G)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('GRN Network Statistics', fontsize=16, fontweight='bold')
        
        # Degree distribution
        degrees = list(degree.values())
        axes[0, 0].hist(degrees, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Degree')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Degree Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # In-degree vs Out-degree
        in_degs = [in_degree[node] for node in G.nodes()]
        out_degs = [out_degree[node] for node in G.nodes()]
        axes[0, 1].scatter(in_degs, out_degs, alpha=0.6, s=50)
        axes[0, 1].set_xlabel('In-Degree')
        axes[0, 1].set_ylabel('Out-Degree')
        axes[0, 1].set_title('In-Degree vs Out-Degree')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Betweenness centrality
        betweenness_vals = list(betweenness.values())
        axes[1, 0].hist(betweenness_vals, bins=30, edgecolor='black', alpha=0.7, color='green')
        axes[1, 0].set_xlabel('Betweenness Centrality')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Betweenness Centrality Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Node type distribution
        node_types = ['TF', 'Target Only', 'Other']
        node_counts = [
            len(tfs),
            len(targets_only),
            G.number_of_nodes() - len(tfs) - len(targets_only)
        ]
        axes[1, 1].bar(node_types, node_counts, color=['yellow', 'lightblue', 'lightgray'], 
                      edgecolor='black', alpha=0.7)
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Node Type Distribution')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        output_file = os.path.join(output_dir, 'grn_statistics_grnndata.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved statistics plot to: {output_file}")
        plt.close()
        
    except Exception as e:
        print(f"Error creating statistics plot: {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Convert GRN CSV to GRnnData format and visualize'
    )
    parser.add_argument('--grn-file', default=DEFAULT_GRN_FILE,
                        help=f'Path to GRN CSV file (default: {DEFAULT_GRN_FILE})')
    parser.add_argument('--min-correlation', type=float, default=MIN_CORRELATION,
                        help=f'Minimum correlation threshold (default: {MIN_CORRELATION})')
    parser.add_argument('--top-edges', type=int, default=TOP_EDGES,
                        help=f'Maximum number of edges (default: {TOP_EDGES})')
    parser.add_argument('--output-dir', default=OUTPUT_DIR,
                        help=f'Output directory (default: {OUTPUT_DIR})')
    
    args = parser.parse_args()
    
    # Convert to GRnnData
    result = convert_to_grnndata(args.grn_file, args.min_correlation, args.top_edges)
    if result is None:
        return
    
    grn_data, grn_filtered = result
    
    # Save GRnnData object
    output_file = os.path.join(args.output_dir, 'grn_heart_failure_grnndata.h5ad')
    save_grnndata(grn_data, output_file)
    
    # Also save the filtered edge list for reference
    edge_list_file = os.path.join(args.output_dir, 'grn_heart_failure_grnndata_edges.csv')
    grn_filtered.to_csv(edge_list_file, index=False)
    print(f"✓ Saved filtered edge list to: {edge_list_file}")
    
    # Visualize
    visualize_grnndata(grn_data, args.output_dir)
    
    print("\n" + "="*60)
    print("Conversion and visualization complete!")
    print("="*60)
    print(f"\nOutput files:")
    print(f"  - GRnnData object: {output_file}")
    print(f"  - Edge list: {edge_list_file}")
    print(f"  - Network visualization: {os.path.join(args.output_dir, 'grn_network_grnndata.png')}")
    print(f"  - Statistics plot: {os.path.join(args.output_dir, 'grn_statistics_grnndata.png')}")


if __name__ == "__main__":
    main()

