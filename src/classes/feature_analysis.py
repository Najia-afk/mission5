import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class FeatureAnalysis:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.features = ['recency_days', 'frequency', 'monetary', 
                        'avg_review_score', 'review_count', 'avg_delivery_time']
    
    def plot_distributions(self):
        """Plot distribution of all features using plotly."""
        fig = make_subplots(rows=3, cols=2, subplot_titles=self.features)
        row, col = 1, 1
        
        for feature in self.features:
            fig.add_trace(
                go.Histogram(x=self.df[feature], name=feature),
                row=row, col=col
            )
            if col == 2:
                row += 1
                col = 1
            else:
                col += 1
        
        fig.update_layout(height=800, title_text="Feature Distributions")
        return fig
    
    def plot_correlation_matrix(self):
        """Plot correlation matrix of features."""
        corr = self.df[self.features].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr,
            x=self.features,
            y=self.features,
            colorscale='RdBu',
            zmin=-1, zmax=1
        ))
        
        fig.update_layout(
            title='Feature Correlation Matrix',
            height=600,
            width=800
        )
        return fig
    
    def plot_feature_boxplots(self):
        """Create boxplots for all features."""
        fig = make_subplots(rows=3, cols=2, subplot_titles=self.features)
        row, col = 1, 1
        
        for feature in self.features:
            fig.add_trace(
                go.Box(y=self.df[feature], name=feature),
                row=row, col=col
            )
            if col == 2:
                row += 1
                col = 1
            else:
                col += 1
                
        fig.update_layout(height=800, title_text="Feature Boxplots")
        return fig