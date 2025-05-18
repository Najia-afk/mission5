import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class FeatureAnalysis:
    def __init__(self, df: pd.DataFrame, columns_to_exclude=None):
        self.df = df
        self.features = list(df.select_dtypes(include=['number']).columns)
        
        # Use the provided columns_to_exclude or default to a list if None
        if columns_to_exclude is None:
            columns_to_exclude = ['customer_id']  # Default columns to exclude
        
        # Filter out excluded columns
        self.features = [col for col in self.features if col not in columns_to_exclude]


    def plot_distributions(self):
        """Plot distribution of all features using plotly."""
        num_features = len(self.features)
        cols = 2  # Number of columns fixed at 2
        rows = (num_features + 1) // 2  # Calculate required number of rows
        
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=self.features)
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