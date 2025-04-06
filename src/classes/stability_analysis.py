import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from typing import Dict, List
from datetime import datetime, timedelta
from tqdm import tqdm

class StabilityAnalysis:
    def __init__(self, df: pd.DataFrame, clustering_model, date_column: str, pca: PCA = None):
        self.df = df.copy()
        self.df[date_column] = pd.to_datetime(self.df[date_column])
        self.model = clustering_model
        self.date_column = date_column
        # Use only the features that were used in clustering
        self.features = [
            'recency_days', 'frequency', 'monetary',
            'avg_review_score', 'review_count',
            'avg_delivery_time'
        ]
        self.periods = None
        self.stability_scores = None
        self.pca = pca

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
            end_idx = start_idx + samples_per_period if i < n_periods - 1 else None
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
            current_features = current[self.features]
            next_features = next_period[self.features]
            
            # Apply PCA if available
            if self.pca is not None:
                current_features = self.pca.transform(current_features)
                next_features = self.pca.transform(next_features)
            
            # Use predict since model is already trained
            labels_current = self.model.predict(current_features)
            labels_next = self.model.predict(next_features)
            
            score = adjusted_rand_score(labels_current, labels_next)
            self.stability_scores[f'{period_names[i]} vs {period_names[i+1]}'] = score
        
        return self.stability_scores

    def find_optimal_periods(self, min_period_days: int = 30, max_periods: int = 12) -> int:
        """Find optimal number of periods based on stability scores and data distribution."""
        total_days = (self.df[self.date_column].max() - self.df[self.date_column].min()).days
        min_samples = len(self.df) * 0.05  # Minimum 5% of data per period
        
        # Start with at least 6 periods or 1 per year
        min_periods = max(6, total_days // 365)
        period_scores = {}
        
        for n_periods in tqdm(range(min_periods, min(max_periods + 1, total_days // min_period_days + 1)),
                            desc="Testing period counts"):
            try:
                # Create periods and check minimum samples
                self.create_time_periods(n_periods)
                if min(len(df) for df in self.periods.values()) < min_samples:
                    continue
                    
                # Compute stability scores
                scores = self.compute_stability_scores()
                avg_stability = np.mean(list(scores.values()))
                std_stability = np.std(list(scores.values()))
                
                # Calculate period length uniformity
                period_lengths = [(df[self.date_column].max() - df[self.date_column].min()).days 
                                for df in self.periods.values()]
                length_variance = np.std(period_lengths) / np.mean(period_lengths)
                
                # Combined score that balances:
                # 1. High average stability (weighted more)
                # 2. Low stability variance
                # 3. Period length uniformity
                # 4. Gentler preference for more granular periods
                period_penalty = 0.5 * (1 / n_periods)
                score = (avg_stability * 1.5 * 
                        (1 - std_stability) * 
                        (1 - length_variance) * 
                        (1 - period_penalty))
                
                period_scores[n_periods] = {
                    'avg_stability': avg_stability,
                    'std_stability': std_stability,
                    'length_variance': length_variance,
                    'combined_score': score
                }
                
            except Exception as e:
                print(f"Error testing {n_periods} periods: {str(e)}")
                continue
        
        if not period_scores:
            raise ValueError("No valid period configurations found")
            
        # Print detailed results
        print("\nPeriod Analysis Results:")
        print(f"{'Periods':<8} {'Stability':<10} {'Std Dev':<10} {'Length Var':<10} {'Score'}")
        print("-" * 60)
        for n_periods, metrics in period_scores.items():
            print(f"{n_periods:<8} {metrics['avg_stability']:.3f}     "
                  f"{metrics['std_stability']:.3f}     "
                  f"{metrics['length_variance']:.3f}     "
                  f"{metrics['combined_score']:.3f}")
        
        # Find optimal periods based on combined score
        optimal_periods = max(period_scores.items(), key=lambda x: x[1]['combined_score'])[0]
        
        # Plot results
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(period_scores.keys()),
            y=[m['avg_stability'] for m in period_scores.values()],
            name='Average Stability',
            mode='lines+markers'
        ))
        fig.add_trace(go.Scatter(
            x=list(period_scores.keys()),
            y=[m['combined_score'] for m in period_scores.values()],
            name='Combined Score',
            mode='lines+markers'
        ))
        
        fig.update_layout(
            title='Period Optimization Results',
            xaxis_title='Number of Periods',
            yaxis_title='Score',
            showlegend=True
        )
        fig.show()
        
        return optimal_periods

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