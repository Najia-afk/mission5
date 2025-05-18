import numpy as np
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import plotly.express as px
from src.classes.feature_transformation import GenericFeatureTransformer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from yellowbrick.features import FeatureImportances
from sklearn.inspection import permutation_importance
import os

# Set environment variable to silence the core detection warning
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

class ClusteringAnalysis:
    def __init__(self, df: pd.DataFrame, features: List[str] = None, transformer: GenericFeatureTransformer = None, n_jobs=4):
        """
        Initialize clustering analysis with transformed data
        
        Parameters:
        -----------
        df : DataFrame
            Transformed feature data
        features : list, optional
            Features to use for clustering (defaults to all numeric columns)
        transformer : GenericFeatureTransformer, optional
            Transformer used to create features, needed for inverse transform
        """
        
        self.n_jobs = n_jobs
        self.df = df
        self.transformer = transformer
        
        # Auto-detect numeric features if not specified
        if features is None:
            self.features = df.select_dtypes(include=['number']).columns.tolist()
        else:
            self.features = features
            
        self.kmeans_results = {}
        
        # Apply PCA with explained variance ratio - changed from 3 to 10 components by default
        # This allows visualization of more components while retaining only 3 for clustering
        self.pca = PCA(n_components=min(10, len(self.features)))
        self.X = self.pca.fit_transform(df[self.features])
        
        # For clustering, we'll still use only the first 3 components
        self.X_cluster = self.X[:, :min(3, len(self.features))]
        
        # Create a consistent color palette (inspired by Yellowbrick)
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
        
    def elbow_method(self, k_range: range) -> Dict:
        """Optimized elbow method using MiniBatchKMeans."""
        print(f"Explained variance ratio of PCA components: {self.pca.explained_variance_ratio_}")
        inertias = []
        silhouette_scores = []
        
        for k in tqdm(k_range, desc="Computing clusters"):
            kmeans = MiniBatchKMeans(
                n_clusters=k, 
                batch_size=1024,
                random_state=42
            )
            kmeans.fit(self.X)
            inertias.append(kmeans.inertia_)
            if k > 1:
                score = silhouette_score(
                    self.X, 
                    kmeans.labels_,
                    sample_size=min(20000, len(self.X))
                )
                silhouette_scores.append(score)
            
        return {
            'k_range': list(k_range),
            'inertia': inertias,
            'silhouette': silhouette_scores
        }
    
    def plot_elbow(self, k_range: range) -> go.Figure:
        """Plot elbow curve with plotly."""
        results = self.elbow_method(k_range)
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Use consistent colors from the palette
        fig.add_trace(
            go.Scatter(x=results['k_range'], y=results['inertia'], 
                        name="Inertia", line=dict(color=self.cluster_colors[0]))
        )
        
        fig.add_trace(
            go.Scatter(x=results['k_range'][1:], y=results['silhouette'],
                        name="Silhouette Score", line=dict(color=self.cluster_colors[1])),
            secondary_y=True
        )
        
        fig.update_layout(
            title='Elbow Method with Silhouette Score',
            xaxis_title='Number of Clusters (k)',
            yaxis_title='Inertia',
            yaxis2_title='Silhouette Score',
            width=900,
            height=600,
            showlegend=True
        )
        
        return fig
    
    def plot_pca_explained_variance(self, max_components: int = 20) -> go.Figure:
        """
        Plot PCA explained variance by number of components.
        
        Parameters:
        -----------
        max_components : int
            Maximum number of components to analyze
        
        Returns:
        --------
        fig : plotly Figure
            Interactive plot of explained variance
        """
        # Limit max_components to number of features
        max_components = min(max_components, len(self.features))
        
        # Fit PCA with max components
        pca = PCA(n_components=max_components)
        pca.fit(self.df[self.features])
        
        # Get explained variance data
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        components = list(range(1, max_components + 1))
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add individual explained variance
        fig.add_trace(
            go.Bar(
                x=components, 
                y=explained_variance,
                name="Individual Explained Variance",
                marker_color='rgb(55, 83, 109)'
            ),
            secondary_y=False
        )
        
        # Add cumulative explained variance
        fig.add_trace(
            go.Scatter(
                x=components, 
                y=cumulative_variance,
                name="Cumulative Explained Variance",
                mode='lines+markers',
                line=dict(color='red', width=2)
            ),
            secondary_y=True
        )
        
        # Update layout
        fig.update_layout(
            title='PCA Explained Variance by Number of Components',
            xaxis_title='Number of Components',
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.5)'),
            width=900,
            height=600
        )
        
        fig.update_yaxes(title_text="Individual Explained Variance", secondary_y=False)
        fig.update_yaxes(title_text="Cumulative Explained Variance", secondary_y=True)
        
        return fig
    
    def plot_pca_biplot(self, n_features: int = 20, include_points: bool = False, sample_size: int = 1000) -> go.Figure:
        """
        Create a PCA biplot showing feature loadings on principal components.
        
        Parameters:
        -----------
        n_features : int
            Number of top features to display by loading magnitude
        include_points : bool
            Whether to include data points (default: False)
        sample_size : int
            Number of observations to sample if include_points=True
            
        Returns:
        --------
        fig : plotly Figure
            Interactive biplot of feature loadings
        """
        # Get PCA loadings (feature weights for each component)
        loadings = self.pca.components_.T
        
        # Only take first 2 components for the biplot
        pc1_loadings = loadings[:, 0]
        pc2_loadings = loadings[:, 1]
        
        # Get feature names
        feature_names = self.features
        
        # Create a DataFrame with loadings
        loadings_df = pd.DataFrame(
            loadings[:, :2],
            columns=['PC1', 'PC2'],
            index=feature_names
        )
        
        # Sort features by magnitude of loading
        loadings_df['magnitude'] = np.sqrt(loadings_df['PC1']**2 + loadings_df['PC2']**2)
        loadings_df = loadings_df.sort_values('magnitude', ascending=False)
        
        # Select top features
        top_features = loadings_df.iloc[:n_features].index.tolist()
        
        # Set up figure
        fig = go.Figure()
        
        # Add scatter plot for feature loadings
        fig.add_trace(
            go.Scatter(
                x=loadings_df.loc[top_features, 'PC1'],
                y=loadings_df.loc[top_features, 'PC2'],
                mode='markers+text',
                text=top_features,
                textposition='top center',
                marker=dict(
                    size=10, 
                    color='rgba(55, 83, 109, 0.7)',
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                name='Feature Loadings'
            )
        )
        
        # Add lines from origin to each point
        for feature in top_features:
            x = loadings_df.loc[feature, 'PC1']
            y = loadings_df.loc[feature, 'PC2']
            fig.add_shape(
                type='line',
                x0=0, y0=0, x1=x, y1=y,
                line=dict(color='rgba(55, 83, 109, 0.3)', width=1)
            )
        
        # Add a circle to represent correlation of 1
        theta = np.linspace(0, 2*np.pi, 100)
        x = np.cos(theta)
        y = np.sin(theta)
        fig.add_trace(
            go.Scatter(
                x=x, y=y,
                mode='lines',
                line=dict(color='rgba(0,0,0,0.3)', width=1),
                showlegend=False
            )
        )
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'PCA Biplot: Feature Loadings on Principal Components',
                'font': {'size': 18, 'family': "Arial, sans-serif"}
            },
            xaxis=dict(
                title=f'PC1 ({self.pca.explained_variance_ratio_[0]:.2%} variance)',
                range=[-1.1, 1.1],
                zeroline=True, 
                zerolinewidth=1, 
                zerolinecolor='black',
                gridcolor='#EEEEEE'
            ),
            yaxis=dict(
                title=f'PC2 ({self.pca.explained_variance_ratio_[1]:.2%} variance)',
                range=[-1.1, 1.1],
                zeroline=True, 
                zerolinewidth=1, 
                zerolinecolor='black',
                gridcolor='#EEEEEE'
            ),
            width=900,
            height=700,
            legend=dict(
                x=0.01, 
                y=0.99,
                bgcolor='rgba(255, 255, 255, 0.8)'
            ),
            template='plotly_white'
        )
        
        return fig
        
    def plot_pca_feature_importance(self, n_components: int = 3) -> go.Figure:
        """
        Visualize the importance of each feature in the principal components.
        
        Parameters:
        -----------
        n_components : int
            Number of components to visualize
            
        Returns:
        --------
        fig : plotly Figure
            Feature importance heatmap
        """
        # Get feature names and loadings
        feature_names = self.features
        n_components = min(n_components, len(self.pca.components_))
        
        # Create a dataframe of loadings
        loadings_df = pd.DataFrame(
            data=self.pca.components_[:n_components, :].T,
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=feature_names
        )
        
        # Take absolute values for importance
        abs_loadings = np.abs(loadings_df)
        
        # Sort by sum of loadings across components
        sorted_idx = abs_loadings.sum(axis=1).sort_values(ascending=False).index
        abs_loadings = abs_loadings.loc[sorted_idx]
        
        # Create heatmap
        fig = px.imshow(
            abs_loadings,
            labels=dict(x="Principal Component", y="Feature", color="Absolute Loading"),
            x=[f'PC{i+1}<br>({self.pca.explained_variance_ratio_[i]:.2%})' for i in range(n_components)],
            y=abs_loadings.index,
            color_continuous_scale="Viridis",
            aspect="auto"
        )
        
        fig.update_layout(
            title='Feature Importance in Principal Components',
            width=900,
            height=max(500, 20 * len(feature_names)),
            coloraxis_colorbar=dict(title="Absolute Loading")
        )
        
        return fig
    
    def plot_kmeans_feature_importance(self, n_clusters: int = None, n_repeats: int = 3, sample_size: int = 100000) -> go.Figure:
        """
        Calculate and visualize feature importance for K-Means clustering using permutation importance.
        
        Parameters:
        -----------
        n_clusters : int, optional
            Number of clusters for K-Means (if None, uses previously fitted model)
        n_repeats : int
            Number of times to permute a feature (lower means faster calculation)
        sample_size : int
            Maximum number of samples to use for faster computation
            
        Returns:
        --------
        fig : plotly Figure
            Feature importance bar chart
        """
        # Always define X_orig first to ensure it's available in all code paths
        X_orig = self.df[self.features].values
        
        # Sample data if it's too large (for faster computation)
        if len(X_orig) > sample_size:
            print(f"Sampling {sample_size} records from {len(X_orig)} for faster calculation...")
            indices = np.random.choice(len(X_orig), sample_size, replace=False)
            X_sample = X_orig[indices]
        else:
            X_sample = X_orig
        
        # Use the existing model if available
        if n_clusters is None and 'orig_model' in self.kmeans_results:
            kmeans_orig = self.kmeans_results['orig_model']
            # Fix: Update n_clusters to the actual number in the model
            n_clusters = kmeans_orig.n_clusters
        else:
            # Create a new model if needed
            if n_clusters is None:
                n_clusters = self.kmeans_results.get('model', {}).n_clusters if 'model' in self.kmeans_results else 4
            
            # Train a KMeans model directly on original features for importance calculation
            kmeans_orig = MiniBatchKMeans(
                n_clusters=n_clusters,
                batch_size=1024,
                random_state=42
            )
            
            print("Fitting MiniBatchKMeans on original features...")
            kmeans_orig.fit(X_orig)
        
        # Define scorer function for silhouette - must accept y even though we don't use it
        def silhouette_scorer(estimator, X, y=None):
            labels = estimator.predict(X)
            # Use much smaller sample for silhouette score calculation
            return silhouette_score(X, labels, sample_size=min(1000, len(X)))
        
        # Calculate permutation importance on original features
        print(f"Calculating feature importance with {n_repeats} repeats (this may take a moment)...")
        result = permutation_importance(
            kmeans_orig, 
            X_sample,  # Use the sampled dataset
            None,  # No target for unsupervised
            scoring=silhouette_scorer,
            n_repeats=n_repeats,
            random_state=42
        )
        print("Feature importance calculation complete!")
        
        # Create DataFrame of results
        importance_df = pd.DataFrame({
            'Feature': self.features,
            'Importance': result.importances_mean,
            'StdDev': result.importances_std
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Create bar chart with consistent colors
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
            title=f'Feature Importance for K-Means Clustering (k={n_clusters})',
            xaxis_title='Silhouette Score Reduction (Higher = More Important)',
            yaxis_title='Feature',
            width=900,
            height=max(500, 20 * len(self.features)),
            coloraxis_showscale=False
        )
        
        return fig
    
    def fit_kmeans(self, n_clusters: int) -> np.ndarray:
        # For visualization - use PCA components (for 3D plots)
        kmeans_pca = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1024, random_state=42)
        self.kmeans_results['labels'] = kmeans_pca.fit_predict(self.X_cluster)  # Use only the first 3 components
        self.kmeans_results['model'] = kmeans_pca
        
        # For feature importance - use original features
        kmeans_orig = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1024, random_state=42)
        X_orig = self.df[self.features].values
        kmeans_orig.fit(X_orig)
        self.kmeans_results['orig_model'] = kmeans_orig
        
        return self.kmeans_results['labels']
    
    def plot_clusters_3d(self, labels: np.ndarray) -> go.Figure:
        """Plot 3D scatter plot of clusters using PCA components."""
        # Create a discrete color map based on our cluster colors
        cluster_colors_dict = {}
        for i in range(max(labels) + 1):
            cluster_colors_dict[i] = self.get_cluster_color(i)
            
        # Map labels to colors
        marker_colors = [cluster_colors_dict.get(l, '#777777') for l in labels]
        
        fig = go.Figure(data=[
            go.Scatter3d(
                x=self.X[:, 0],  # First PCA component
                y=self.X[:, 1],  # Second PCA component
                z=self.X[:, 2],  # Third PCA component
                mode='markers',
                marker=dict(
                    size=5,
                    color=marker_colors,
                ),
                text=[f'{self.get_cluster_name(l)}' for l in labels]
            )
        ])
        
        fig.update_layout(
            title='3D Cluster Visualization (PCA Components)',
            scene=dict(
                xaxis_title='PC1',
                yaxis_title='PC2',
                zaxis_title='PC3'
            ),
            width=900,
            height=900
        )
        
        return fig
    
    def plot_silhouette_plotly(self, n_clusters: int = None, figsize: tuple = (900, 600)) -> go.Figure:
        """
        Create a silhouette plot to visualize cluster quality using Plotly.
        The height of each cluster section will accurately reflect the relative cluster size.
        """
        # Get number of clusters
        if n_clusters is None:
            if 'model' in self.kmeans_results:
                n_clusters = self.kmeans_results['model'].n_clusters
            else:
                n_clusters = 3  # Default
        
        # Get actual cluster sizes from the complete dataset
        if 'labels' in self.kmeans_results and len(self.kmeans_results['labels']) == len(self.df):
            # Use existing labels if they match the dataset size
            full_labels = self.kmeans_results['labels']
            full_cluster_counts = np.bincount(full_labels)
            full_cluster_proportions = full_cluster_counts / full_cluster_counts.sum()
        else:
            # Otherwise, run on the full dataset to get true proportions
            print("Computing cluster proportions on full dataset...")
            kmeans_full = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1024, random_state=42)
            full_labels = kmeans_full.fit_predict(self.df[self.features].values)
            full_cluster_counts = np.bincount(full_labels)
            full_cluster_proportions = full_cluster_counts / full_cluster_counts.sum()
        
        # Sample data if needed for faster silhouette calculation
        X_sample = self.df[self.features].values
        if len(X_sample) > 20000:
            print(f"Sampling 20,000 records from {len(X_sample)} for faster silhouette calculation...")
            
            # Stratified sampling to maintain cluster proportions
            sample_size = 20000
            indices = []
            
            for i in range(n_clusters):
                # Find indices for this cluster
                cluster_indices = np.where(full_labels == i)[0]
                
                # Calculate how many samples to take from this cluster
                # to maintain the same proportion as in the full dataset
                n_samples = int(sample_size * full_cluster_proportions[i])
                if n_samples > 0:  # Ensure we take at least some samples
                    cluster_sample = np.random.choice(cluster_indices, 
                                                      size=min(n_samples, len(cluster_indices)), 
                                                      replace=False)
                    indices.extend(cluster_sample)
            
            # If we didn't get enough samples (due to rounding), add more randomly
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
        from sklearn.metrics import silhouette_samples
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
            y_lower = y_upper + 5  # Less spacing between clusters
        
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
            title=f'Silhouette Analysis for KMeans Clustering (k={n_clusters})',
            xaxis_title='Silhouette Coefficient',
            yaxis_title='Cluster Distribution',
            width=figsize[0],
            height=figsize[1],
            showlegend=True,
            xaxis=dict(range=[-0.1, 1.05]),  # Silhouette values range from -1 to 1
            yaxis=dict(showticklabels=False)  # Hide y-axis tick labels
        )
        
        return fig
    
    def plot_intercluster_distance_circles(self, n_clusters: int = None, figsize: tuple = (900, 700)) -> go.Figure:
        """
        Create a circle-based visualization showing relationships between cluster centers.
        """
        from scipy.spatial.distance import pdist, squareform
        from sklearn.manifold import MDS
        import numpy as np
        
        # Get number of clusters
        if n_clusters is None:
            if 'model' in self.kmeans_results:
                n_clusters = self.kmeans_results['model'].n_clusters
            else:
                n_clusters = 3  # Default
        
        # Get cluster centers and sizes
        if 'orig_model' in self.kmeans_results and (n_clusters is None or 
                n_clusters == self.kmeans_results['orig_model'].n_clusters):
            centers = self.kmeans_results['orig_model'].cluster_centers_
            labels = self.kmeans_results.get('labels', None)
        else:
            # Create and fit a new model
            print(f"Fitting KMeans with {n_clusters} clusters...")
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1024, random_state=42)
            X = self.df[self.features].values
            labels = kmeans.fit_predict(X)
            centers = kmeans.cluster_centers_
        
        # Get cluster sizes if labels are available
        if labels is not None:
            cluster_sizes = np.bincount(labels)
            size_scale = 100  # Max circle size
            sizes = (cluster_sizes / cluster_sizes.max()) * size_scale
        else:
            sizes = np.ones(n_clusters) * 50  # Default size if no labels
        
        # Compute pairwise distances between centers
        distances = pdist(centers)
        distance_matrix = squareform(distances)
        
        # Use MDS to position clusters in 2D space based on their distances
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        positions = mds.fit_transform(distance_matrix)
        
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
                text=[self.get_cluster_name(i)],
                hoverinfo='text'
            ))
        
        # Add lines between clusters with distance labels in the middle of the line
        max_dist = np.max(distances)
        for i in range(n_clusters):
            for j in range(i+1, n_clusters):
                dist = distance_matrix[i, j]
                normalized_dist = dist/max_dist
                
                # Calculate line points including the midpoint for text placement
                line_color = self.get_cluster_color(i)
                # Width based on proximity - thicker for closer clusters
                width = 1.5 + 3 * (1 - normalized_dist)
                
                # Add a line with distance shown in the middle in the same color as the line
                fig.add_trace(go.Scatter(
                    x=[positions[i, 0], positions[j, 0]],
                    y=[positions[i, 1], positions[j, 1]],
                    mode='lines+text',
                    text=["", ""],  # Empty text elements at the ends
                    textposition='middle center',
                    line=dict(
                        color=line_color,
                        width=width,
                        dash='solid' if normalized_dist < 0.5 else 'dot'  # Solid for close, dotted for far
                    ),
                    hoverinfo='text',
                    hovertext=[f'Distance: {dist:.2f}'],
                    showlegend=False
                ))
                
                # Add a separate trace just for the text at the midpoint
                mid_x = (positions[i, 0] + positions[j, 0]) / 2
                mid_y = (positions[i, 1] + positions[j, 1]) / 2
                
                fig.add_trace(go.Scatter(
                    x=[mid_x],
                    y=[mid_y],
                    mode='text',
                    text=[f"{dist:.2f}"],
                    textposition='middle center',
                    textfont=dict(
                        color=line_color,  # Match the line color
                        size=10,
                        family='Arial'
                    ),
                    hoverinfo='none',
                    showlegend=False
                ))
        
        # Update layout
        fig.update_layout(
            title="Intercluster Relationship Visualization",
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