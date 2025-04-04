import numpy as np
import plotly.graph_objects as go
from typing import List, Tuple, Dict
import pandas as pd

class CorrelationAnalysis:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numerical_features = [
            'recency_days', 'frequency', 'monetary',
            'avg_review_score', 'review_count', 'negative_reviews',
            'unique_categories', 'avg_delivery_time', 'unique_sellers'
        ]
        self.corr_matrix = None
    
    def compute_correlation(self) -> np.ndarray:
        """Compute correlation matrix."""
        self.corr_matrix = self.df[self.numerical_features].corr()
        return self.corr_matrix
    
    def get_lower_triangle(self) -> np.ndarray:
        """Return lower triangle of correlation matrix."""
        if self.corr_matrix is None:
            self.compute_correlation()
        mask = np.triu(np.ones_like(self.corr_matrix))
        return self.corr_matrix * mask
    
    def plot_correlation_matrix(self) -> go.Figure:
        """Generate correlation matrix heatmap showing only lower triangle."""
        if self.corr_matrix is None:
            self.get_lower_triangle()
            
        # Create mask for upper triangle (including diagonal)
        mask = np.triu(np.ones_like(self.corr_matrix), k=1)  # k=1 excludes diagonal
        # Apply inverse mask to correlation matrix (keep lower triangle)
        masked_corr = np.ma.masked_array(self.corr_matrix, mask)
        
        fig = go.Figure(data=go.Heatmap(
            z=masked_corr,
            x=self.numerical_features,
            y=self.numerical_features,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            text=np.round(masked_corr, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False,
            showscale=True
        ))
        
        fig.update_layout(
            title='Feature Correlation Analysis - Lower Triangle',
            xaxis={'tickangle': 45},
            yaxis={'tickangle': 0}
        )
        
        return fig