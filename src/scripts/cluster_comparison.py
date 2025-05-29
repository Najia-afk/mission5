import pandas as pd
import plotly.express as px
import numpy as np
from typing import List, Dict, Optional, Union, Tuple

def compare_clustering_methods(
    labels1: np.ndarray, 
    labels2: np.ndarray, 
    method1_name: str = "Method 1", 
    method2_name: str = "Method 2",
    normalize: str = "index",
    color_scale: str = "Viridis",
    figsize: Tuple[int, int] = (900, 500),
    index_prefix: str = "Cluster",
    include_noise: bool = True
) -> px.imshow:
    """
    Compare two clustering methods by creating a heatmap showing the overlap between clusters.
    
    Parameters:
    -----------
    labels1 : numpy.ndarray
        Cluster labels from first method
    labels2 : numpy.ndarray
        Cluster labels from second method
    method1_name : str
        Name of the first clustering method (y-axis)
    method2_name : str
        Name of the second clustering method (x-axis)
    normalize : str
        How to normalize the data: 'index' (rows), 'columns', or 'all'
    color_scale : str
        Plotly color scale for the heatmap
    figsize : tuple
        Figure dimensions (width, height)
    index_prefix : str
        Prefix for cluster labels (default: "Cluster")
    include_noise : bool
        Whether to include noise points (-1 labels) in the comparison
        
    Returns:
    --------
    fig : plotly.express.imshow
        Heatmap visualization of cluster overlap
    """
    # Create a mapping for labels to their string representation
    def get_cluster_label(i: int, prefix: str = index_prefix) -> str:
        if i == -1:
            return "Noise"
        return f"{prefix} {i}"
    
    # Create a DataFrame for cross-tabulation
    comparison_data = pd.DataFrame({
        'method1': labels1,
        'method2': labels2
    })
    
    # Create cross-tabulation
    cross_tab = pd.crosstab(
        comparison_data['method1'], 
        comparison_data['method2'],
        normalize=normalize
    )
    
    # Get unique labels for each method
    unique_labels1 = sorted(np.unique(labels1))
    unique_labels2 = sorted(np.unique(labels2))
    
    # Remove noise points if not included
    if not include_noise:
        if -1 in unique_labels1:
            unique_labels1.remove(-1)
        if -1 in unique_labels2:
            unique_labels2.remove(-1)
    
    # Create axis labels
    x_labels = [get_cluster_label(i) for i in cross_tab.columns]
    y_labels = [get_cluster_label(i) for i in cross_tab.index]
    
    # Create the heatmap
    fig = px.imshow(
        cross_tab,
        labels=dict(x=method2_name, y=method1_name, color="Proportion"),
        x=x_labels,
        y=y_labels,
        color_continuous_scale=color_scale,
        title=f"Overlap Between {method1_name} and {method2_name} Clusters"
    )
    
    # Update layout
    fig.update_layout(
        width=figsize[0], 
        height=figsize[1],
        coloraxis_colorbar=dict(
            title="Proportion",
            tickformat=".1%"
        )
    )
    
    return fig

def create_multi_comparison_dashboard(
    labels_dict: Dict[str, np.ndarray],
    normalize: str = "index",
    color_scale: str = "Viridis",
    figsize: Tuple[int, int] = (900, 500)
) -> Dict[str, px.imshow]:
    """
    Create multiple comparison visualizations between different clustering methods.
    
    Parameters:
    -----------
    labels_dict : dict
        Dictionary mapping method names to their cluster labels
    normalize : str
        How to normalize the data: 'index', 'columns', or 'all'
    color_scale : str
        Plotly color scale for the heatmap
    figsize : tuple
        Figure dimensions (width, height)
        
    Returns:
    --------
    figs : dict
        Dictionary of figures for each comparison
    """
    methods = list(labels_dict.keys())
    n_methods = len(methods)
    figures = {}
    
    # Create pairwise comparisons
    for i in range(n_methods):
        for j in range(i+1, n_methods):
            method1 = methods[i]
            method2 = methods[j]
            
            # Check if labels exist for both methods
            if labels_dict[method1] is not None and labels_dict[method2] is not None:
                # Create comparison
                fig = compare_clustering_methods(
                    labels_dict[method1],
                    labels_dict[method2],
                    method1_name=method1,
                    method2_name=method2,
                    normalize=normalize,
                    color_scale=color_scale,
                    figsize=figsize
                )
                
                # Store figure
                figures[f"{method1}_vs_{method2}"] = fig
    
    return figures
