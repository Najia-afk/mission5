import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.express as px
from tqdm.notebook import tqdm
import random
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ClusterStabilityAnalysis:
    """Analyze and visualize the stability of clustering results."""
    
    def __init__(self, df, transformer, original_df_with_dates=None):
        """
        Initialize the stability analysis class.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The transformed dataframe used for clustering
        transformer : GenericFeatureTransformer
            The transformer used to process the data
        original_df_with_dates : pandas.DataFrame, optional
            Original dataframe containing temporal information
        """
        self.df = df
        self.transformer = transformer
        self.original_df_with_dates = original_df_with_dates
        
    def evaluate_bootstrap_stability(self, n_clusters=4, n_iterations=20, sample_fraction=0.8, random_state=42):
        """
        Evaluate cluster stability using bootstrap sampling.
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters to create
        n_iterations : int
            Number of bootstrap iterations
        sample_fraction : float
            Fraction of data to sample in each iteration
        random_state : int
            Random seed for reproducibility
        
        Returns:
        --------
        dict
            Dictionary containing stability metrics and visualizations
        """
        np.random.seed(random_state)
        sample_size = int(len(self.df) * sample_fraction)
        
        # Store cluster assignments for each iteration
        all_labels = []
        all_indices = []
        
        # Run clustering on bootstrap samples
        for i in tqdm(range(n_iterations), desc="Bootstrap Iterations"):
            # Sample indices with replacement
            indices = np.random.choice(self.df.index, size=sample_size, replace=False)
            all_indices.append(set(indices))
            
            # Get the sample
            sample = self.df.loc[indices]
            
            # Cluster the sample
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            labels = kmeans.fit_predict(sample)
            
            # Store labels with their indices
            labels_dict = {idx: label for idx, label in zip(indices, labels)}
            all_labels.append(labels_dict)
        
        # Calculate ARI between all pairs of iterations
        ari_scores = []
        iteration_pairs = []
        
        for i in range(n_iterations):
            for j in range(i+1, n_iterations):
                # Find common indices
                common_indices = list(all_indices[i].intersection(all_indices[j]))
                
                if len(common_indices) < 10:  # Need enough samples for meaningful comparison
                    continue
                    
                # Get labels for common indices
                labels_i = np.array([all_labels[i][idx] for idx in common_indices])
                labels_j = np.array([all_labels[j][idx] for idx in common_indices])
                
                # Calculate ARI
                ari = adjusted_rand_score(labels_i, labels_j)
                ari_scores.append(ari)
                iteration_pairs.append((i, j))
        
        # Create visualization
        fig = go.Figure()
        
        # Add ARI scores
        fig.add_trace(go.Scatter(
            x=list(range(len(ari_scores))),
            y=ari_scores,
            mode='markers',
            name='ARI Score',
            marker=dict(
                size=8,
                color=ari_scores,
                colorscale='Viridis',
                colorbar=dict(title='ARI Score'),
                showscale=True
            ),
            text=[f"Iterations {i+1} & {j+1}: ARI={ari:.3f}" for (i, j), ari in zip(iteration_pairs, ari_scores)]
        ))
        
        # Add average line
        avg_ari = np.mean(ari_scores)
        fig.add_trace(go.Scatter(
            x=[0, len(ari_scores)-1],
            y=[avg_ari, avg_ari],
            mode='lines',
            name=f'Average ARI: {avg_ari:.3f}',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title='Cluster Stability: Adjusted Rand Index (ARI) Across Bootstrap Samples',
            xaxis_title='Iteration Pair Index',
            yaxis_title='Adjusted Rand Index',
            yaxis_range=[-0.05, 1.05],
            template='plotly_white',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        return {
            'ari_scores': ari_scores,
            'mean_ari': avg_ari,
            'std_ari': np.std(ari_scores),
            'figure': fig
        }
    
    def evaluate_temporal_stability(self, n_clusters=4, period='quarter', min_orders=1, min_customers_per_period=None, random_state=42):
        """
        Evaluate cluster stability across time periods.
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters to create
        period : str
            Time period to use ('month', 'quarter', 'year')
        min_orders : int
            Minimum number of orders a customer must have to be included
        min_customers_per_period : int, optional
            Minimum number of customers required per period (defaults to n_clusters * 5 if None)
        random_state : int
            Random seed for reproducibility
        
        Returns:
        --------
        dict
            Dictionary containing stability metrics and visualizations
        """
        if self.original_df_with_dates is None:
            raise ValueError("Original dataframe with dates must be provided for temporal stability analysis")
        
        # Set minimum customers threshold
        if min_customers_per_period is None:
            min_customers_per_period = n_clusters * 5
        
        # Ensure order_purchase_timestamp is in datetime format
        date_col = 'order_purchase_timestamp'
        df_dates = self.original_df_with_dates.reset_index()
        
        if date_col not in df_dates.columns:
            raise ValueError(f"Column {date_col} not found in original dataframe")
        
        # Convert to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df_dates[date_col]):
            df_dates[date_col] = pd.to_datetime(df_dates[date_col])
        
        # Create period column
        if period == 'month':
            df_dates['period'] = df_dates[date_col].dt.to_period('M')
        elif period == 'quarter':
            df_dates['period'] = df_dates[date_col].dt.to_period('Q')
        elif period == 'year':
            df_dates['period'] = df_dates[date_col].dt.to_period('Y')
        else:
            raise ValueError("Period must be 'month', 'quarter', or 'year'")
        
        # Get unique periods
        periods = sorted(df_dates['period'].unique())
        
        # Store cluster assignments for each period
        period_clusters = {}
        period_customer_counts = {}
        
        # Get unique customer IDs
        all_customers = set(df_dates['customer_id'])
        
        # Run clustering for each period
        for p in tqdm(periods, desc=f"Clustering by {period}"):
            # Get customers for this period
            period_customers = df_dates[df_dates['period'] == p]['customer_id'].unique()
            period_customer_counts[p] = len(period_customers)
            
            if len(period_customers) < min_customers_per_period:
                print(f"Skipping period {p} - insufficient customers ({len(period_customers)})")
                continue
                
            # Get transformed data for these customers
            if all(c in self.df.index for c in period_customers):
                period_data = self.df.loc[period_customers]
                
                # Cluster the period data
                kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
                labels = kmeans.fit_predict(period_data)
                
                # Store labels with customer IDs
                period_clusters[p] = {cust: label for cust, label in zip(period_customers, labels)}
        
        # Calculate ARI between consecutive periods
        ari_scores = []
        period_pairs = []
        
        sorted_periods = sorted(period_clusters.keys())
        for i in range(len(sorted_periods) - 1):
            p1 = sorted_periods[i]
            p2 = sorted_periods[i + 1]
            
            # Find common customers
            common_customers = list(set(period_clusters[p1].keys()) & set(period_clusters[p2].keys()))
            
            if len(common_customers) < min_customers_per_period:
                print(f"Skipping {p1} to {p2} comparison - insufficient common customers ({len(common_customers)})")
                continue
                
            # Get labels for common customers
            labels_p1 = np.array([period_clusters[p1][cust] for cust in common_customers])
            labels_p2 = np.array([period_clusters[p2][cust] for cust in common_customers])
            
            # Calculate ARI
            ari = adjusted_rand_score(labels_p1, labels_p2)
            ari_scores.append(ari)
            period_pairs.append((p1, p2))
        
        # Initialize migration matrices
        migration_matrices = {}
        
        # Calculate customer migration between clusters
        if len(sorted_periods) >= 2:
            for i in range(len(sorted_periods) - 1):
                p1 = sorted_periods[i]
                p2 = sorted_periods[i + 1]
                
                # Find common customers
                common_customers = list(set(period_clusters[p1].keys()) & set(period_clusters[p2].keys()))
                
                if len(common_customers) < 10:
                    continue
                
                # Create migration matrix
                migration_matrix = np.zeros((n_clusters, n_clusters))
                
                for cust in common_customers:
                    from_cluster = period_clusters[p1][cust]
                    to_cluster = period_clusters[p2][cust]
                    migration_matrix[from_cluster, to_cluster] += 1
                
                # Convert to percentages
                row_sums = migration_matrix.sum(axis=1, keepdims=True)
                migration_matrix_pct = np.divide(migration_matrix, row_sums, 
                                            out=np.zeros_like(migration_matrix), 
                                            where=row_sums!=0) * 100
                
                migration_matrices[(p1, p2)] = migration_matrix_pct
        
        # Create visualizations
        fig_ari = go.Figure()
        
        if ari_scores:
            # Add ARI scores
            period_labels = [f"{p1} to {p2}" for p1, p2 in period_pairs]
            fig_ari.add_trace(go.Scatter(
                x=period_labels,
                y=ari_scores,
                mode='lines+markers',
                name='ARI Score',
                marker=dict(
                    size=10,
                    color=ari_scores,
                    colorscale='Viridis',
                    colorbar=dict(title='ARI Score'),
                    showscale=True
                ),
                text=[f"{p1} to {p2}: ARI={ari:.3f}" for (p1, p2), ari in zip(period_pairs, ari_scores)]
            ))
            
            # Add average line
            avg_ari = np.mean(ari_scores)
            fig_ari.add_trace(go.Scatter(
                x=[period_labels[0], period_labels[-1]],
                y=[avg_ari, avg_ari],
                mode='lines',
                name=f'Average ARI: {avg_ari:.3f}',
                line=dict(color='red', dash='dash')
            ))
            
            fig_ari.update_layout(
                title=f'Cluster Stability: ARI Between Consecutive {period.capitalize()} Periods',
                xaxis_title=f'{period.capitalize()} Transition',
                yaxis_title='Adjusted Rand Index',
                yaxis_range=[-0.05, 1.05],
                template='plotly_white',
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
        
        # Create customer count by period visualization
        fig_counts = go.Figure()
        
        if period_customer_counts:
            periods_str = [str(p) for p in period_customer_counts.keys()]
            counts = list(period_customer_counts.values())
            
            fig_counts.add_trace(go.Bar(
                x=periods_str,
                y=counts,
                marker_color='lightblue',
                text=counts,
                textposition='outside'
            ))
            
            fig_counts.update_layout(
                title=f'Customer Count by {period.capitalize()}',
                xaxis_title=period.capitalize(),
                yaxis_title='Number of Customers',
                template='plotly_white'
            )
        
        # Create migration matrices visualizations
        migration_figs = {}
        
        for (p1, p2), matrix in migration_matrices.items():
            fig_migration = go.Figure(data=go.Heatmap(
                z=matrix,
                x=[f'Cluster {i}' for i in range(n_clusters)],
                y=[f'Cluster {i}' for i in range(n_clusters)],
                colorscale='Viridis',
                text=[[f"{v:.1f}%" for v in row] for row in matrix],
                texttemplate="%{text}",
                colorbar=dict(title='Percentage')
            ))
            
            fig_migration.update_layout(
                title=f'Cluster Migration from {p1} to {p2}',
                xaxis_title=f'Cluster in {p2} (To)',
                yaxis_title=f'Cluster in {p1} (From)',
                template='plotly_white'
            )
            
            migration_figs[(p1, p2)] = fig_migration
        
        return {
            'ari_scores': ari_scores,
            'period_pairs': period_pairs,
            'mean_ari': np.mean(ari_scores) if ari_scores else None,
            'std_ari': np.std(ari_scores) if ari_scores else None,
            'period_customer_counts': period_customer_counts,
            'migration_matrices': migration_matrices,
            'figure_ari': fig_ari,
            'figure_counts': fig_counts,
            'migration_figures': migration_figs
        }

    def evaluate_cross_period_stability(self, n_clusters=4, period='year', eval_sample_size=1000, min_customers_per_period=None, random_state=42):
        """
        Evaluate cluster stability across time periods using a common evaluation set.
        This method works even when there's no customer overlap between periods.
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters to create
        period : str
            Time period to use ('month', 'quarter', 'year')
        eval_sample_size : int
            Size of evaluation sample from largest period
        min_customers_per_period : int, optional
            Minimum number of customers required per period (defaults to n_clusters * 5 if None)
        random_state : int
            Random seed for reproducibility
        
        Returns:
        --------
        dict
            Dictionary containing stability metrics and visualizations
        """
        if self.original_df_with_dates is None:
            raise ValueError("Original dataframe with dates must be provided for temporal stability analysis")
        
        # Set random seed
        np.random.seed(random_state)
        
        # Set minimum customers threshold
        if min_customers_per_period is None:
            min_customers_per_period = n_clusters * 5
            
        # Ensure order_purchase_timestamp is in datetime format
        date_col = 'order_purchase_timestamp'
        df_dates = self.original_df_with_dates.reset_index()
        
        if date_col not in df_dates.columns:
            raise ValueError(f"Column {date_col} not found in original dataframe")
        
        # Convert to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df_dates[date_col]):
            df_dates[date_col] = pd.to_datetime(df_dates[date_col])
        
        # Create period column
        if period == 'month':
            df_dates['period'] = df_dates[date_col].dt.to_period('M')
            df_dates['period_type'] = 'month'
        elif period == 'quarter':
            df_dates['period'] = df_dates[date_col].dt.to_period('Q')
            df_dates['period_type'] = 'quarter'
        elif period == 'year':
            df_dates['period'] = df_dates[date_col].dt.year
            df_dates['period_type'] = 'year'
        else:
            raise ValueError("Period must be 'month', 'quarter', or 'year'")
            
        # Get period counts
        period_counts = df_dates['period'].value_counts()
        #print(f"Customer counts by {period}:")
        #print(period_counts.sort_index())
            
        # Train a model for each period
        period_models = {}
        period_customer_counts = {}
        
        for p in tqdm(period_counts.index, desc=f"Training models by {period}"):
            # Get customers for this period
            period_customers = df_dates[df_dates['period'] == p]['customer_id'].unique()
            period_customer_counts[p] = len(period_customers)
            
            if len(period_customers) < min_customers_per_period:
                print(f"Skipping period {p} - insufficient customers ({len(period_customers)})")
                continue
                
            # Verify these customers exist in the transformed data
            valid_customers = [c for c in period_customers if c in self.df.index]
            if len(valid_customers) < min_customers_per_period:
                print(f"Skipping period {p} - insufficient customers with features ({len(valid_customers)})")
                continue
                
            # Get transformed data for these customers
            period_data = self.df.loc[valid_customers]
            
            # Train clustering model
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            kmeans.fit(period_data)
            
            # Store model
            period_models[p] = kmeans
            
        # Create evaluation set from largest period
        if len(period_models) == 0:
            return {
                'error': 'No periods had sufficient customers for modeling',
                'period_customer_counts': period_customer_counts
            }
            
        # Find largest period with a model
        valid_periods = [p for p in period_models.keys()]
        largest_period = max(valid_periods, key=lambda p: period_customer_counts[p])
        
        # Sample customers from largest period
        largest_period_customers = df_dates[df_dates['period'] == largest_period]['customer_id'].unique()
        valid_customers = [c for c in largest_period_customers if c in self.df.index]
        
        sample_size = min(eval_sample_size, len(valid_customers))
        eval_customers = np.random.choice(valid_customers, size=sample_size, replace=False)
        
        # Get feature data for evaluation
        eval_features = self.df.loc[eval_customers]
        
        # Get predictions from each period's model
        period_predictions = {}
        for p, model in period_models.items():
            period_predictions[p] = model.predict(eval_features)
            
        # Calculate ARI between all period pairs
        ari_scores = []
        period_pairs = []
        
        periods = sorted(period_predictions.keys())
        for i in range(len(periods)):
            for j in range(i+1, len(periods)):
                p_i = periods[i]
                p_j = periods[j]
                labels_i = period_predictions[p_i]
                labels_j = period_predictions[p_j]
                
                ari = adjusted_rand_score(labels_i, labels_j)
                ari_scores.append(ari)
                period_pairs.append((p_i, p_j))
                #print(f"ARI between {p_i} and {p_j}: {ari:.4f}")
                
        # Create stability visualization
        fig = go.Figure()
        
        # Add bar chart of ARI values
        pair_labels = [f"{p1} vs {p2}" for p1, p2 in period_pairs]
        fig.add_trace(go.Bar(
            x=pair_labels,
            y=ari_scores,
            marker_color='lightblue',
            text=[f"{ari:.3f}" for ari in ari_scores],
            textposition='outside'
        ))
        
        # Calculate mean ARI if we have scores
        if ari_scores:
            mean_ari = np.mean(ari_scores)
            # Add mean line
            fig.add_trace(go.Scatter(
                x=[pair_labels[0], pair_labels[-1]],
                y=[mean_ari, mean_ari],
                mode='lines',
                name=f'Mean ARI: {mean_ari:.3f}',
                line=dict(color='red', dash='dash')
            ))
        else:
            mean_ari = None
            
        fig.update_layout(
            title=f'Cluster Stability Between {period.capitalize()}s (Using Common Evaluation Set)',
            xaxis_title=f'{period.capitalize()} Comparison',
            yaxis_title='Adjusted Rand Index',
            yaxis_range=[0, 1],
            template='plotly_white',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        # Create period counts visualization
        fig_counts = go.Figure()
        
        if period_customer_counts:
            periods_str = [str(p) for p in period_customer_counts.keys()]
            counts = list(period_customer_counts.values())
            
            fig_counts.add_trace(go.Bar(
                x=periods_str,
                y=counts,
                marker_color='lightblue',
                text=counts,
                textposition='outside'
            ))
            
            fig_counts.update_layout(
                title=f'Customer Count by {period.capitalize()}',
                xaxis_title=period.capitalize(),
                yaxis_title='Number of Customers',
                template='plotly_white'
            )
        
        return {
            'ari_scores': ari_scores,
            'period_pairs': period_pairs,
            'mean_ari': mean_ari,
            'std_ari': np.std(ari_scores) if ari_scores else None,
            'period_customer_counts': period_customer_counts,
            'figure': fig,
            'figure_counts': fig_counts,
            'eval_sample_size': sample_size,
            'largest_period': largest_period
        }