import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple
from tqdm import tqdm
import plotly.express as px
from src.classes.feature_transformation import FeatureTransformation

class ClusteringAnalysis:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.features = [
            'recency_days', 'frequency', 'monetary',
            'avg_review_score', 'review_count',
            'avg_delivery_time'
        ]
        self.kmeans_results = {}
        # Apply PCA with explained variance ratio
        self.pca = PCA(n_components=3)
        self.X = self.pca.fit_transform(df[self.features])
        
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
                    sample_size=min(10000, len(self.X))
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
        
        # Add inertia trace
        fig.add_trace(
            go.Scatter(x=results['k_range'], y=results['inertia'], 
                      name="Inertia", line=dict(color='blue'))
        )
        
        # Add silhouette score trace
        fig.add_trace(
            go.Scatter(x=results['k_range'][1:], y=results['silhouette'],
                      name="Silhouette Score", line=dict(color='red')),
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
    
    def fit_kmeans(self, n_clusters: int) -> np.ndarray:
        """Fit MiniBatchKMeans clustering."""
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=1024,
            random_state=42
        )
        self.kmeans_results['labels'] = kmeans.fit_predict(self.X)
        self.kmeans_results['model'] = kmeans
        return self.kmeans_results['labels']
    
    def plot_clusters_3d(self, labels: np.ndarray) -> go.Figure:
        """Plot 3D scatter plot of clusters using PCA components."""            
        fig = go.Figure(data=[
            go.Scatter3d(
                x=self.X[:, 0],  # First PCA component
                y=self.X[:, 1],  # Second PCA component
                z=self.X[:, 2],  # Third PCA component
                mode='markers',
                marker=dict(
                    size=5,
                    color=labels,
                    colorscale='Viridis',
                ),
                text=[f'Cluster {l}' for l in labels]
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
    
    def analyze_clusters(self, labels: np.ndarray) -> Dict:
        """Analyze cluster characteristics."""
        # Add PCA variance explanation
        pca_explained = {
            f"PC{i+1}": var for i, var in 
            enumerate(self.pca.explained_variance_ratio_)
        }
        
        # Get cluster sizes
        cluster_sizes = pd.Series(labels).value_counts().to_dict()
        
        # Get cluster means for original features
        cluster_means = pd.DataFrame(self.df[self.features])
        cluster_means['Cluster'] = labels
        cluster_means = cluster_means.groupby('Cluster').mean()
        
        return {
            'pca_explained': pca_explained,
            'cluster_sizes': cluster_sizes,
            'cluster_means': cluster_means
        }

    def plot_cluster_analysis(self, analysis_results: Dict) -> go.Figure:
        """Plot cluster analysis results."""
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'PCA Explained Variance',
                'Cluster Sizes',
                'Feature Means by Cluster'
            )
        )
        
        # Plot PCA explained variance
        fig.add_trace(
            go.Bar(
                x=list(analysis_results['pca_explained'].keys()),
                y=list(analysis_results['pca_explained'].values()),
                name='Explained Variance'
            ),
            row=1, col=1
        )
        
        # Plot cluster sizes
        fig.add_trace(
            go.Bar(
                x=list(analysis_results['cluster_sizes'].keys()),
                y=list(analysis_results['cluster_sizes'].values()),
                name='Cluster Sizes'
            ),
            row=1, col=2
        )
        
        # Plot feature means
        for feature in self.features:
            fig.add_trace(
                go.Scatter(
                    x=analysis_results['cluster_means'].index,
                    y=analysis_results['cluster_means'][feature],
                    name=feature
                ),
                row=2, col=1
            )
        
        fig.update_layout(height=800, width=1200, showlegend=True)
        return fig
    
    def analyze_clusters_business(self, labels: np.ndarray) -> Dict:
        """Analyze cluster characteristics in business terms."""
        # Get cluster means for original features
        cluster_means = pd.DataFrame(
            self.df[self.features], 
            columns=self.features
        )
        cluster_means['Cluster'] = labels
        profile = cluster_means.groupby('Cluster').mean()
        
        # Use the same FeatureTransformation instance
        ft = FeatureTransformation()  # This will return the existing instance with fitted scalers
        profile_original = ft.inverse_transform_features(profile)
        
        # Create cluster descriptions
        cluster_profiles = {}
        for cluster in profile_original.index:
            row = profile_original.loc[cluster]
            
            # Use original scale values for thresholds
            recency = "Recent" if row['recency_days'] < 30 else "Inactive"
            frequency = "Frequent" if row['frequency'] > 2 else "Occasional"
            spending = "High-value" if row['monetary'] > profile_original['monetary'].mean() else "Standard-value"
            satisfaction = "Satisfied" if row['avg_review_score'] >= 4 else "Needs Attention"
            
            # Create cluster profile
            cluster_profiles[f"Cluster {cluster}"] = {
                "Size": len(cluster_means[cluster_means['Cluster'] == cluster]),
                "Type": f"{frequency} {spending} Customers",
                "Activity": recency,
                "Satisfaction": satisfaction,
                "Key Metrics": {
                    "Avg Order Value": f"${row['monetary']/max(1, row['frequency']):.0f}",  # Prevent division by zero
                    "Days Since Last Purchase": f"{row['recency_days']:.0f}",
                    "Total Orders": f"{row['frequency']:.1f}",
                    "Review Score": f"{row['avg_review_score']:.1f}/5",
                    "Delivery Time (days)": f"{row['avg_delivery_time']:.1f}"
                }
            }
        
        return {
            'cluster_profiles': cluster_profiles,
            'cluster_means': profile_original,
            'cluster_sizes': pd.Series(labels).value_counts().to_dict()
        }
    
    def plot_cluster_analysis_business(self, analysis_results: Dict) -> go.Figure:
        """Plot business-friendly cluster profiles with consistent colors per cluster."""
        profiles = analysis_results['cluster_profiles']
        
        # Define a consistent color palette for clusters
        colors = px.colors.qualitative.Set3[:len(profiles)]  # or any other color palette
        
        # Create subplot figure with specific types
        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{"type": "domain"}, {"type": "xy"}],  # First row
                [{"type": "xy"}, {"type": "xy"}]       # Second row
            ],
            subplot_titles=(
                'Customer Segments Distribution',
                'Average Order Value by Segment',
                'Customer Satisfaction by Segment',
                'Customer Activity Patterns'
            )
        )
        
        # Plot cluster sizes (pie chart)
        sizes = [p['Size'] for p in profiles.values()]
        labels = [f"{k}: {p['Type']}" for k, p in profiles.items()]
        
        fig.add_trace(
            go.Pie(
                labels=labels, 
                values=sizes, 
                name='Segment Sizes',
                marker=dict(colors=colors)
            ),
            row=1, col=1
        )
        
        cluster_keys = list(profiles.keys())
        
        # Plot average order values (bar chart)
        order_values = [float(p['Key Metrics']['Avg Order Value'].replace('$','')) 
                    for p in profiles.values()]
        
        fig.add_trace(
            go.Bar(
                x=cluster_keys, 
                y=order_values, 
                name='Avg Order Value',
                marker_color=colors
            ),
            row=1, col=2
        )
        
        # Plot satisfaction scores (bar chart)
        satisfaction = [float(p['Key Metrics']['Review Score'].split('/')[0]) 
                    for p in profiles.values()]
        
        fig.add_trace(
            go.Bar(
                x=cluster_keys, 
                y=satisfaction, 
                name='Satisfaction',
                marker_color=colors
            ),
            row=2, col=1
        )
        
        # Plot recency (bar chart)
        recency = [float(p['Key Metrics']['Days Since Last Purchase']) 
                for p in profiles.values()]
        
        fig.add_trace(
            go.Bar(
                x=cluster_keys, 
                y=recency, 
                name='Days Since Purchase',
                marker_color=colors
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="Customer Segment Analysis",
            showlegend=True,
            # Ensure consistent appearance
            bargap=0.3,
            bargroupgap=0.1
        )
        
        return fig