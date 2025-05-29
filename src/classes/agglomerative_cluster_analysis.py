import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Tuple, Optional
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import io
import base64
from sklearn.neighbors import NearestCentroid
import warnings
import hashlib
import pickle
import os
from tqdm.notebook import tqdm

warnings.filterwarnings('ignore')

class AgglomerativeClusterAnalysis:
    def __init__(self, df: pd.DataFrame, features: List[str] = None, transformer = None, pca_components: np.ndarray = None):
        """
        Initialize Agglomerative Clustering analysis
        
        Parameters:
        -----------
        df : DataFrame
            Feature data for clustering
        features : list, optional
            Features to use for clustering (defaults to all numeric columns)
        transformer : object, optional
            Transformer with inverse_transform method
        pca_components : numpy array, optional
            PCA components to use for visualization (if provided)
        """
        self.df = df
        self.transformer = transformer
        
        # Auto-detect numeric features if not specified
        if features is None:
            self.features = df.select_dtypes(include=['number']).columns.tolist()
        else:
            self.features = features
            
        # Store PCA components if provided
        self.pca_components = pca_components
        
        # Results storage
        self.clustering_results = {}
        
        # Create a consistent color palette
        self.cluster_colors = [
            '#5cb85c',  # green
            '#5bc0de',  # blue
            '#f0ad4e',  # orange
            '#d9534f',  # red
            '#9370DB',  # purple
            '#C71585',  # magenta
            '#20B2AA',  # teal
            '#F08080',  # coral
            '#4682B4',  # steel blue
            '#FFD700',  # gold
        ]
        
        # Define transparent versions for fill areas
        self.cluster_colors_transparent = [
            f'rgba(92, 184, 92, 0.3)',    # green
            f'rgba(91, 192, 222, 0.3)',   # blue
            f'rgba(240, 173, 78, 0.3)',   # orange
            f'rgba(217, 83, 79, 0.3)',    # red
            f'rgba(147, 112, 219, 0.3)',  # purple
            f'rgba(199, 21, 133, 0.3)',   # magenta
            f'rgba(32, 178, 170, 0.3)',   # teal
            f'rgba(240, 128, 128, 0.3)',  # coral
            f'rgba(70, 130, 180, 0.3)',   # steel blue
            f'rgba(255, 215, 0, 0.3)',    # gold
        ]
    
    def get_cluster_name(self, cluster_idx: int) -> str:
        """Return a consistent name for a cluster index"""
        return f'Cluster {cluster_idx}'
    
    def get_cluster_color(self, cluster_idx: int, transparent: bool = False) -> str:
        """Return a consistent color for a cluster index"""
        color_list = self.cluster_colors_transparent if transparent else self.cluster_colors
        return color_list[cluster_idx % len(color_list)]
    
    def plot_dendrogram(self, max_samples: int = 1000, figsize: tuple = (900, 500)) -> go.Figure:
        """
        Create a dendrogram visualization of hierarchical clustering
        
        Parameters:
        -----------
        max_samples : int
            Maximum number of samples to use for dendrogram (for speed)
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        fig : plotly Figure
            Dendrogram visualization
        """
        # Decide what data to use
        if self.pca_components is not None:
            X = self.pca_components[:, :3]  # Use first 3 PCA components
            print(f"Using PCA components for dendrogram")
        else:
            X = self.df[self.features].values
            print(f"Using original features for dendrogram")
        
        # Sample data if it's too large (for faster computation)
        if len(X) > max_samples:
            print(f"Sampling {max_samples} records from {len(X)} for faster dendrogram...")
            indices = np.random.choice(len(X), max_samples, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        # Compute the linkage matrix
        Z = linkage(X_sample, method='ward')
        
        # Create matplotlib figure for dendrogram
        plt_fig, ax = plt.subplots(figsize=(figsize[0]/100, figsize[1]/100), dpi=100)
        dendrogram(Z, ax=ax, leaf_rotation=90)
        ax.set_title('Hierarchical Clustering Dendrogram')
        ax.set_xlabel('Sample index')
        ax.set_ylabel('Distance')
        
        # Convert matplotlib figure to plotly
        img_bytes = io.BytesIO()
        plt_fig.savefig(img_bytes, format='png', bbox_inches='tight')
        img_bytes.seek(0)
        img_base64 = base64.b64encode(img_bytes.read()).decode('ascii')
        plt.close(plt_fig)  # Close matplotlib figure
        
        # Create plotly figure with image
        fig = go.Figure()
        
        fig.add_layout_image(
            dict(
                source=f'data:image/png;base64,{img_base64}',
                x=0,
                y=1,
                xref="paper",
                yref="paper",
                sizex=1,
                sizey=1,
                sizing="stretch",
                layer="below"
            )
        )
        
        # Update layout
        fig.update_layout(
            width=figsize[0],
            height=figsize[1],
            margin=dict(l=20, r=20, t=60, b=20),
        )
        
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        
        return fig
    
    def fit_clusters(self, n_clusters: int, linkage: str = 'ward', affinity: str = 'euclidean', sample_size: int = None) -> np.ndarray:
        """
        Fit Agglomerative Clustering model
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters to create
        linkage : str
            Linkage criterion ('ward', 'complete', 'average', 'single')
        affinity : str
            Metric used to compute linkage ('euclidean', 'l1', 'l2', etc.)
        sample_size : int, optional
            Maximum number of samples to use for fitting (if None, use all data)
            
        Returns:
        --------
        labels : numpy array
            Cluster labels for each data point
        """
        # Check cache for existing results
        cache_key = f"agglom_{n_clusters}_{linkage}_{affinity}_{sample_size}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        cache_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache', f'{cache_hash}.pkl')
        
        # Try to load from cache
        if os.path.exists(cache_file):
            try:
                print(f"Loading clustering results from cache...")
                with open(cache_file, 'rb') as f:
                    cached_results = pickle.load(f)
                self.clustering_results = cached_results
                return cached_results['labels']
            except Exception as e:
                print(f"Error loading from cache: {e}")
        
        # For feature importance - always use original features
        X_orig = self.df[self.features].values
        
        # Sample data if it's too large and sample_size is specified
        if sample_size is not None and len(X_orig) > sample_size:
            print(f"Sampling {sample_size} records from {len(X_orig)} for faster clustering...")
            indices = np.random.choice(len(X_orig), sample_size, replace=False)
            X_sample = X_orig[indices]
            
            # Fit on sample
            if linkage == 'ward':
                # Ward linkage only works with euclidean distance
                model = AgglomerativeClustering(
                    n_clusters=n_clusters, 
                    linkage=linkage
                )
            else:
                # For other linkage methods, use metric
                model = AgglomerativeClustering(
                    n_clusters=n_clusters, 
                    linkage=linkage,
                    metric=affinity
                )
            sample_labels = model.fit_predict(X_sample)
            
            # Instead of retraining on full dataset, use nearest centroid approach
            print("Predicting clusters for full dataset using nearest centroids...")
            
            # Calculate centroids of each cluster in the sample
            centroids = []
            for i in range(n_clusters):
                mask = sample_labels == i
                if np.any(mask):
                    cluster_points = X_sample[mask]
                    centroids.append(np.mean(cluster_points, axis=0))
                else:
                    # Handle empty clusters (shouldn't happen often)
                    centroids.append(np.zeros(X_sample.shape[1]))
            
            # Use nearest centroid classifier to predict remaining points
            centroid_classifier = NearestCentroid()
            centroid_classifier.fit(X_sample, sample_labels)
            labels = centroid_classifier.predict(X_orig)
            
            # Store model and results
            self.clustering_results = {
                'model': model,
                'labels': labels,
                'n_clusters': n_clusters,
                'linkage': linkage,
                'affinity': affinity,
                'centroids': centroids,
                'sample_indices': indices,
                'sample_labels': sample_labels
            }
        else:
            # If not using sampling or sample size equals full dataset
            if sample_size is None and len(X_orig) > 10000:
                warnings.warn(
                    f"Fitting Agglomerative Clustering on {len(X_orig)} samples without sampling. "
                    f"This may require significant memory and time. Consider using sample_size parameter."
                )
            
            # Fit on full dataset
            if linkage == 'ward':
                model = AgglomerativeClustering(
                    n_clusters=n_clusters, 
                    linkage=linkage
                )
            else:
                model = AgglomerativeClustering(
                    n_clusters=n_clusters, 
                    linkage=linkage,
                    metric=affinity
                )
            labels = model.fit_predict(X_orig)
            
            # Store results
            self.clustering_results = {
                'model': model,
                'labels': labels,
                'n_clusters': n_clusters,
                'linkage': linkage,
                'affinity': affinity
            }
        
        # Cache results
        try:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(self.clustering_results, f)
            print(f"Cached clustering results to {cache_file}")
        except Exception as e:
            print(f"Error caching results: {e}")
        
        return labels
    
    def plot_silhouette(self, n_clusters: int = None, figsize: tuple = (900, 600), sample_size: int = 20000) -> go.Figure:
        """
        Create a silhouette plot to visualize cluster quality using Plotly.
        The height of each cluster section will accurately reflect the relative cluster size.
        
        Parameters:
        -----------
        n_clusters : int, optional
            Number of clusters (if None, uses previously fitted model)
        figsize : tuple
            Figure size (width, height)
        sample_size : int
            Maximum number of samples to use for silhouette calculation
            
        Returns:
        --------
        fig : plotly Figure
            Silhouette plot visualization
        """
        # Get number of clusters
        if n_clusters is None:
            if 'n_clusters' in self.clustering_results:
                n_clusters = self.clustering_results['n_clusters']
            else:
                n_clusters = 3  # Default
        
        # Get or compute labels
        if 'labels' in self.clustering_results and self.clustering_results['n_clusters'] == n_clusters:
            full_labels = self.clustering_results['labels']
        else:
            # Fit new model if needed
            print(f"Fitting Agglomerative Clustering with {n_clusters} clusters...")
            full_labels = self.fit_clusters(n_clusters)
            
        # Get cluster counts and proportions
        full_cluster_counts = np.bincount(full_labels)
        full_cluster_proportions = full_cluster_counts / full_cluster_counts.sum()
        
        # Sample data if needed for faster silhouette calculation
        X_sample = self.df[self.features].values
        if len(X_sample) > sample_size:
            print(f"Sampling {sample_size} records from {len(X_sample)} for faster silhouette calculation...")
            
            # Stratified sampling to maintain cluster proportions
            indices = []
            
            for i in range(n_clusters):
                # Find indices for this cluster
                cluster_indices = np.where(full_labels == i)[0]
                
                # Calculate how many samples to take from this cluster
                n_samples = int(sample_size * full_cluster_proportions[i])
                if n_samples > 0:  # Ensure we take at least some samples
                    cluster_sample = np.random.choice(cluster_indices, 
                                                      size=min(n_samples, len(cluster_indices)), 
                                                      replace=False)
                    indices.extend(cluster_sample)
            
            # If we didn't get enough samples, add more randomly
            if len(indices) < sample_size:
                remaining = sample_size - len(indices)
                all_indices = set(range(len(X_sample)))
                remaining_indices = list(all_indices - set(indices))
                if remaining_indices:
                    extra_indices = np.random.choice(remaining_indices, 
                                                   size=min(remaining, len(remaining_indices)), 
                                                   replace=False)
                    indices.extend(extra_indices)
            
            X_sample = X_sample[indices]
            sample_labels = full_labels[indices]
        else:
            sample_labels = full_labels
        
        # Calculate silhouette scores
        silhouette_vals = silhouette_samples(X_sample, sample_labels)
        
        # Calculate average silhouette score
        avg_score = np.mean(silhouette_vals)
        
        # Create a DataFrame for visualization
        silhouette_df = pd.DataFrame({
            'sample_idx': range(len(silhouette_vals)),
            'cluster': sample_labels,
            'silhouette_val': silhouette_vals
        })
        
        # Sort within each cluster for better visualization
        silhouette_df = silhouette_df.sort_values(['cluster', 'silhouette_val'])
        
        # Create figure
        fig = go.Figure()
        
        # Add silhouette traces for each cluster
        # Scale the total height based on figure size
        total_height = figsize[1] * 0.8  # 80% of figure height for the plots
        
        # Starting position
        y_lower = 10
        
        for i in range(n_clusters):
            # Get silhouette values for current cluster
            cluster_silhouette_vals = silhouette_df[silhouette_df['cluster'] == i]['silhouette_val']
            cluster_silhouette_vals = cluster_silhouette_vals.sort_values()
            
            if len(cluster_silhouette_vals) == 0:
                continue  # Skip empty clusters
                
            # Calculate height based on proportion
            cluster_height = total_height * full_cluster_proportions[i]
            
            # Calculate y positions
            y_upper = y_lower + cluster_height
            y_positions = np.linspace(y_lower, y_upper - 1, len(cluster_silhouette_vals))
            
            # Use consistent colors
            fill_color = self.get_cluster_color(i, transparent=True)
            line_color = self.get_cluster_color(i)
            
            # Add the silhouette plot for this cluster
            fig.add_trace(
                go.Scatter(
                    x=cluster_silhouette_vals,
                    y=y_positions,
                    mode='lines',
                    line=dict(width=0.5, color=line_color),
                    fill='tozerox',
                    fillcolor=fill_color,
                    name=f"{self.get_cluster_name(i)} ({full_cluster_counts[i]} samples, {full_cluster_proportions[i]:.1%})"
                )
            )
            
            # Update y_lower for next cluster
            y_lower = y_upper + 5
        
        # Add a vertical line for the average silhouette score
        fig.add_vline(
            x=avg_score, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Avg Silhouette: {avg_score:.3f}",
            annotation_position="top right"
        )
        
        # Update layout
        fig.update_layout(
            title=f'Silhouette Analysis for Agglomerative Clustering (k={n_clusters})',
            xaxis_title='Silhouette Coefficient',
            yaxis_title='Cluster Distribution',
            width=figsize[0],
            height=figsize[1],
            showlegend=True,
            xaxis=dict(range=[-0.1, 1.05]),
            yaxis=dict(showticklabels=False)
        )
        
        return fig
    
    def compare_linkage_methods(self, n_clusters: int = 4, sample_size: int = 5000) -> go.Figure:
        """
        Compare different linkage methods for hierarchical clustering
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters to create
        sample_size : int
            Number of samples to use for faster computation
            
        Returns:
        --------
        fig : plotly Figure
            Comparison of linkage methods
        """
        # Sample data if needed
        X = self.df[self.features].values
        if len(X) > sample_size:
            indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        # Linkage methods to compare
        linkage_methods = ['ward', 'complete', 'average', 'single']
        
        # Results storage
        results = []
        
        # Run clustering with each linkage method
        for method in tqdm(linkage_methods, desc="Comparing linkage methods"):
            # Note: Ward linkage only works with euclidean distance
            if method == 'ward':
                model = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage=method
                )
            else:
                model = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage=method,
                    metric='euclidean'  # Use metric instead of affinity for non-ward methods
                )
                
            labels = model.fit_predict(X_sample)
            
            # Calculate silhouette score
            if len(np.unique(labels)) > 1:  # Only calculate if more than one cluster
                score = silhouette_score(X_sample, labels)
            else:
                score = 0
                
            # Store results
            results.append({
                'method': method,
                'silhouette': score,
                'n_clusters_found': len(np.unique(labels)),
                'largest_cluster': np.max(np.bincount(labels)) / len(labels)
            })
        
        # Create DataFrame
        results_df = pd.DataFrame(results)
        
        # Create figure
        fig = make_subplots(rows=1, cols=2, 
                           subplot_titles=('Silhouette Score by Linkage Method', 
                                          'Largest Cluster Proportion'))
        
        # Add silhouette score bars
        fig.add_trace(
            go.Bar(
                x=results_df['method'],
                y=results_df['silhouette'],
                name='Silhouette Score',
                marker_color=self.cluster_colors[0]
            ),
            row=1, col=1
        )
        
        # Add largest cluster proportion bars
        fig.add_trace(
            go.Bar(
                x=results_df['method'],
                y=results_df['largest_cluster'],
                name='Largest Cluster Proportion',
                marker_color=self.cluster_colors[1]
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f'Comparison of Linkage Methods (k={n_clusters})',
            width=900,
            height=500,
            showlegend=True,
        )
        
        return fig
    
    def plot_intercluster_distance(self, n_clusters: int = None, figsize: tuple = (900, 700)) -> go.Figure:
        """
        Create a circle-based visualization showing relationships between cluster centers.
        
        Parameters:
        -----------
        n_clusters : int, optional
            Number of clusters (if None, uses previously fitted model)
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        fig : plotly Figure
            Visualization of cluster relationships
        """
        from scipy.spatial.distance import pdist, squareform
        from sklearn.manifold import MDS
        
        # Get number of clusters
        if n_clusters is None:
            if 'n_clusters' in self.clustering_results:
                n_clusters = self.clustering_results['n_clusters']
            else:
                n_clusters = 4  # Default
        
        # Get or compute labels
        if 'labels' in self.clustering_results and self.clustering_results['n_clusters'] == n_clusters:
            labels = self.clustering_results['labels']
        else:
            # Fit new model if needed
            print(f"Fitting Agglomerative Clustering with {n_clusters} clusters...")
            labels = self.fit_clusters(n_clusters)
        
        # Decide what data to use
        if self.pca_components is not None:
            X = self.pca_components[:, :3]  # Use first 3 PCA components
        else:
            X = self.df[self.features].values
        
        # Calculate cluster representatives (mean of each cluster)
        representatives = []
        cluster_sizes = []
        
        for cluster_id in range(n_clusters):
            cluster_points = X[labels == cluster_id]
            if len(cluster_points) > 0:  # Ensure cluster is not empty
                representatives.append(np.mean(cluster_points, axis=0))
                cluster_sizes.append(len(cluster_points))
            else:
                # Handle empty clusters (should be rare)
                representatives.append(np.zeros(X.shape[1]))
                cluster_sizes.append(0)
        
        representatives = np.array(representatives)
        
        # Compute pairwise distances between representatives
        distances = pdist(representatives)
        distance_matrix = squareform(distances)
        
        # Use MDS to position clusters in 2D space based on their distances
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        positions = mds.fit_transform(distance_matrix)
        
        # Scale sizes based on cluster sizes
        max_size = max(cluster_sizes)
        size_scale = 100  # Max circle size
        sizes = [size / max_size * size_scale for size in cluster_sizes]
        
        # Create figure
        fig = go.Figure()
        
        # Add circles for each cluster
        for i in range(n_clusters):
            fig.add_trace(go.Scatter(
                x=[positions[i, 0]],
                y=[positions[i, 1]],
                mode='markers',
                marker=dict(
                    size=sizes[i],
                    color=self.get_cluster_color(i),
                    line=dict(width=2, color='DarkSlateGrey')
                ),
                name=self.get_cluster_name(i),
                text=[f"{self.get_cluster_name(i)}<br>{cluster_sizes[i]} points"],
                hoverinfo='text'
            ))
        
        # Add lines between clusters with distance labels
        max_dist = np.max(distances)
        for i in range(n_clusters):
            for j in range(i+1, n_clusters):
                # Calculate line opacity based on inverse of distance
                opacity = 0.8 * (1 - distance_matrix[i, j] / max_dist) + 0.2
                
                # Add a line connecting the clusters
                fig.add_trace(go.Scatter(
                    x=[positions[i, 0], positions[j, 0]],
                    y=[positions[i, 1], positions[j, 1]],
                    mode='lines',
                    line=dict(
                        width=1,
                        color=f'rgba(100, 100, 100, {opacity:.2f})'
                    ),
                    hoverinfo='text',
                    text=f"Distance: {distance_matrix[i, j]:.3f}",
                    showlegend=False
                ))
        
        # Update layout
        fig.update_layout(
            title=f"Agglomerative Clustering Intercluster Relationship Visualization (k={n_clusters})",
            width=figsize[0],
            height=figsize[1],
            showlegend=True,
            hovermode='closest',
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),
            plot_bgcolor='rgba(240, 240, 240, 0.5)'
        )
        
        return fig
    
    def plot_feature_importance(self, n_clusters: int = None, n_repeats: int = 3, sample_size: int = 10000) -> go.Figure:
        """
        Calculate and visualize feature importance for Agglomerative clustering using permutation importance.
        
        Parameters:
        -----------
        n_clusters : int, optional
            Number of clusters (if None, uses previously fitted model)
        n_repeats : int
            Number of times to permute a feature (lower means faster calculation)
        sample_size : int
            Maximum number of samples to use for faster computation
            
        Returns:
        --------
        fig : plotly Figure
            Feature importance bar chart
        """
        from sklearn.inspection import permutation_importance
        
        # Get number of clusters
        if n_clusters is None:
            if 'n_clusters' in self.clustering_results:
                n_clusters = self.clustering_results['n_clusters']
            else:
                n_clusters = 4  # Default
        
        # Always define X_orig first to ensure it's available in all code paths
        X_orig = self.df[self.features].values
        
        # Sample data if it's too large (for faster computation)
        if len(X_orig) > sample_size:
            print(f"Sampling {sample_size} records from {len(X_orig)} for faster calculation...")
            indices = np.random.choice(len(X_orig), sample_size, replace=False)
            X_sample = X_orig[indices]
        else:
            X_sample = X_orig
        
        # Setup AgglomerativeClustering model with the right parameters
        if 'linkage' in self.clustering_results:
            linkage = self.clustering_results['linkage']
        else:
            linkage = 'ward'  # Default
            
        if linkage == 'ward':
            model = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=linkage
            )
        else:
            affinity = self.clustering_results.get('affinity', 'euclidean')
            model = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=linkage,
                metric=affinity
            )
        
        # Create a scorer for Agglomerative Clustering based on silhouette score
        def agglom_scorer(estimator, X, y=None):
            # Predict clusters
            labels = estimator.fit_predict(X)
            
            # If all points are in one cluster, return 0
            if len(np.unique(labels)) <= 1:
                return 0
                
            # Calculate silhouette score using a sample for efficiency
            try:
                return silhouette_score(
                    X, 
                    labels,
                    sample_size=min(1000, len(X))
                )
            except:
                return 0
        
        # Calculate permutation importance
        print(f"Calculating feature importance with {n_repeats} repeats (this may take a moment)...")
        try:
            result = permutation_importance(
                model, 
                X_sample,
                None,  # No target for unsupervised
                scoring=agglom_scorer,
                n_repeats=n_repeats,
                random_state=42
            )
            print("Feature importance calculation complete!")
        except Exception as e:
            print(f"Error in feature importance calculation: {e}")
            fig = go.Figure()
            fig.add_annotation(
                text=f"Could not calculate feature importance: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="red")
            )
            fig.update_layout(
                title="Feature Importance for Agglomerative Clustering",
                width=900,
                height=500
            )
            return fig
        
        # Create DataFrame of results
        importance_df = pd.DataFrame({
            'Feature': self.features,
            'Importance': result.importances_mean,
            'StdDev': result.importances_std
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Create bar chart
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            error_x='StdDev',
            orientation='h',
            color='Importance',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            title=f'Feature Importance for Agglomerative Clustering (k={n_clusters}, linkage={linkage})',
            xaxis_title='Silhouette Score Reduction (Higher = More Important)',
            yaxis_title='Feature',
            width=900,
            height=max(500, 20 * len(self.features)),
            coloraxis_showscale=False
        )
        
        return fig
