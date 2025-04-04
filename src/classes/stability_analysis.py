import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
import plotly.graph_objects as go
from typing import Dict, List
from datetime import datetime, timedelta
from tqdm import tqdm

class StabilityAnalysis:
    def __init__(self, df: pd.DataFrame, clustering_model, date_column: str):
        self.df = df.copy()
        self.df[date_column] = pd.to_datetime(self.df[date_column])  # Convert to datetime
        self.model = clustering_model
        self.date_column = date_column
        self.numeric_features = [
            'recency_days', 'frequency', 'monetary',
            'avg_review_score', 'review_count', 'avg_delivery_time'
        ]
        self.periods = None
        self.stability_scores = None

    def validate_features(self):
        """Validate numeric features exist in dataframe."""
        missing = [col for col in self.numeric_features if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")

    def create_time_periods(self, n_periods: int = 3) -> Dict[str, pd.DataFrame]:
        """Split data into time periods with equal samples."""
        # Sort by date
        self.df = self.df.sort_values(self.date_column)
        
        # Calculate period boundaries
        total_samples = len(self.df)
        samples_per_period = total_samples // n_periods
        
        self.periods = {}
        for i in range(n_periods):
            start_idx = i * samples_per_period
            end_idx = start_idx + samples_per_period
            self.periods[f'Period {i+1}'] = self.df.iloc[start_idx:end_idx].copy()
        
        return self.periods

    def compute_stability_scores(self) -> Dict[str, float]:
        """Compute stability between consecutive periods."""
        if not self.periods:
            raise ValueError("Call create_time_periods first")
        
        self.stability_scores = {}
        period_names = list(self.periods.keys())
        
        for i in range(len(period_names)-1):
            current = self.periods[period_names[i]]
            next_period = self.periods[period_names[i+1]]
            
            # Ensure equal sample sizes
            min_samples = min(len(current), len(next_period))
            current = current.head(min_samples)
            next_period = next_period.head(min_samples)
            
            # Get features
            current_features = current[self.numeric_features]
            next_features = next_period[self.numeric_features]
            
            # Compute clusters and stability
            labels_current = self.model.fit_predict(current_features)
            labels_next = self.model.fit_predict(next_features)
            
            score = adjusted_rand_score(labels_current, labels_next)
            self.stability_scores[f'{period_names[i]} vs {period_names[i+1]}'] = score
        
        return self.stability_scores

    def plot_stability(self) -> go.Figure:
        """Visualize stability scores."""
        if not self.stability_scores:
            raise ValueError("Call compute_stability_scores first")
            
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(self.stability_scores.keys()),
            y=list(self.stability_scores.values()),
            mode='lines+markers',
            name='Stability Score'
        ))
        
        fig.update_layout(
            title='Cluster Stability Over Time',
            xaxis_title='Time Periods',
            yaxis_title='Adjusted Rand Index (0-1)',
            yaxis=dict(range=[0, 1]),
            width=900,
            height=500,
            showlegend=True
        )
        
        return fig