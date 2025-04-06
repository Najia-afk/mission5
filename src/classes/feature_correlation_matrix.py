import numpy as np
import plotly.graph_objects as go
from typing import List, Tuple, Dict
import pandas as pd

class CorrelationAnalysis:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
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
            self.compute_correlation()
        
        # Create mask for upper triangle (including diagonal)
        mask = np.triu(np.ones_like(self.corr_matrix, dtype=bool))  # Upper triangle mask
        masked_corr = self.corr_matrix.mask(mask)  # Mask upper triangle

        fig = go.Figure(data=go.Heatmap(
            z=masked_corr.values,
            x=self.numerical_features,
            y=self.numerical_features,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            text=np.round(masked_corr.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False,
            showscale=True
        ))
        
        fig.update_layout(
            title='Feature Correlation Analysis - Lower Triangle',
            xaxis={'tickangle': 45},
            yaxis={'tickangle': 0},
            xaxis_title="Features",
            yaxis_title="Features"
        )
        
        return fig