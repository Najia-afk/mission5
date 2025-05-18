import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Union
from src.classes.feature_transformation import GenericFeatureTransformer

# Default configuration for segment naming thresholds
DEFAULT_SEGMENT_CONFIG = {
    # Recency thresholds (in days)
    'recency': {
        'active': 30,      # 0-30 days: "Active" 
        'recent': 90,      # 31-90 days: "Recent"
        # > 90 days: "Inactive"
    },
    
    # Frequency thresholds (number of orders)
    'frequency': {
        'frequent': 3,     # > 3 orders: "Frequent"
        'returning': 1.5,  # > 1.5 orders: "Returning"
        # <= 1.5 orders: "One-time"
    },
    
    # Monetary thresholds
    'monetary': {
        'high_value': 'mean',  # > mean monetary: "High-value"
        # <= mean monetary: "Standard-value"
    },
    
    # Satisfaction thresholds (review scores)
    'satisfaction': {
        'very_satisfied': 4.5,  # >= 4.5: "Very Satisfied"
        'satisfied': 4.0,       # >= 4.0: "Satisfied"
        'neutral': 3.0,         # >= 3.0: "Neutral"
        # < 3.0: "Unsatisfied"
    }
}

class ClusterDashboard:
    """
    Creates an interactive dashboard with cluster insights for sales teams.
    
    This dashboard provides actionable information about customer segments
    including key metrics, sales opportunities, and segment naming.
    """
    
    def __init__(self, cluster_labels: np.ndarray, 
                 feature_data: pd.DataFrame,
                 transformer: Optional[GenericFeatureTransformer] = None,
                 segment_config: Optional[Dict] = None):
        """
        Initialize the dashboard with cluster labels and feature data.
        
        Parameters:
        -----------
        cluster_labels : numpy.ndarray
            Array of cluster assignments for each customer
        feature_data : pandas.DataFrame
            DataFrame containing customer features (transformed or original)
        transformer : GenericFeatureTransformer, optional
            Transformer to convert features back to original scale
        segment_config : Dict, optional
            Dictionary containing threshold configurations for segment naming
        """
        self.labels = cluster_labels
        self.data = feature_data
        self.transformer = transformer
        self.segment_config = segment_config or DEFAULT_SEGMENT_CONFIG
        
        # Store customer IDs if available in the index
        self.has_customer_ids = isinstance(feature_data.index, pd.Index) and not isinstance(feature_data.index, pd.RangeIndex)
        
        # Set a color palette for consistent visualization
        self.cluster_colors = px.colors.qualitative.Safe
        
        # Prepare data and compute statistics
        self.prepare_data()
        
    def prepare_data(self):
        """Prepare the data for analysis and visualization."""
        # Create a DataFrame with cluster labels
        if self.has_customer_ids:
            self.clustered_data = self.data.copy()
            self.clustered_data['cluster'] = self.labels
        else:
            self.clustered_data = self.data.copy()
            self.clustered_data['cluster'] = self.labels
        
        # Get numeric and categorical features
        self.numeric_features = self.data.select_dtypes(include=['number']).columns.tolist()
        self.categorical_features = self.data.select_dtypes(exclude=['number']).columns.tolist()
        
        # Compute cluster statistics
        self._compute_cluster_statistics()
        
        # Generate cluster profiles
        self._generate_cluster_profiles()
        
    def _compute_cluster_statistics(self):
        """Calculate statistics for each cluster."""
        # Group by cluster and calculate statistics
        self.cluster_stats = {}
        
        # Get unique clusters and ensure the clustered_data has values
        unique_clusters = self.clustered_data['cluster'].unique()
        
        if len(unique_clusters) == 0 or len(self.numeric_features) == 0:
            print("Warning: No clusters or numeric features found.")
            # Initialize empty stats dictionaries to prevent KeyError
            for stat_func in ['mean', 'median', 'min', 'max', 'std']:
                self.cluster_stats[stat_func] = pd.DataFrame()
            
            # Calculate cluster sizes even if empty
            self.cluster_sizes = pd.Series(dtype='int64')
            return
        
        # Numeric features - calculate mean, median, min, max, std
        for stat_func in ['mean', 'median', 'min', 'max', 'std']:
            try:
                # Handle case where numeric features might be empty
                if len(self.numeric_features) > 0:
                    # Handle potential errors in groupby aggregation
                    df_stat = self.clustered_data.groupby('cluster')[self.numeric_features].agg(stat_func)
                    self.cluster_stats[stat_func] = df_stat
                else:
                    # Create empty DataFrame with proper index to prevent KeyError
                    self.cluster_stats[stat_func] = pd.DataFrame(index=unique_clusters)
            except Exception as e:
                print(f"Warning: Error calculating {stat_func} statistics: {str(e)}")
                # Create empty DataFrame with proper index to prevent KeyError
                self.cluster_stats[stat_func] = pd.DataFrame(index=unique_clusters)
        
        # Categorical features - calculate mode and frequency with error handling
        if self.categorical_features:
            try:
                cat_stats = {}
                for cluster in unique_clusters:
                    cluster_data = self.clustered_data[self.clustered_data['cluster'] == cluster]
                    cat_stats[cluster] = {}
                    
                    for feature in self.categorical_features:
                        # Skip if feature is not in the data
                        if feature not in cluster_data.columns:
                            continue
                            
                        # Calculate mode and frequency
                        value_counts = cluster_data[feature].value_counts()
                        if not value_counts.empty:
                            mode_value = value_counts.index[0]
                            mode_freq = value_counts.iloc[0] / len(cluster_data)
                            cat_stats[cluster][feature] = {'mode': mode_value, 'frequency': mode_freq}
                        
                self.cluster_stats['categorical'] = cat_stats
            except Exception as e:
                print(f"Warning: Error calculating categorical statistics: {str(e)}")
                self.cluster_stats['categorical'] = {}
        
        # Calculate cluster sizes
        try:
            self.cluster_sizes = self.clustered_data['cluster'].value_counts().sort_index()
        except Exception as e:
            print(f"Warning: Error calculating cluster sizes: {str(e)}")
            self.cluster_sizes = pd.Series(index=unique_clusters, data=0)
    
    def _generate_cluster_profile_name(self, cluster_id: int) -> str:
        """
        Generate a descriptive name for each cluster based on key characteristics.
        
        Parameters:
        -----------
        cluster_id : int
            Cluster identifier
            
        Returns:
        --------
        str
            Descriptive name for the cluster
        """
        # Get original scale stats if transformer is available
        if self.transformer is not None:
            try:
                # Convert the stats back to original scale
                mean_stats = self.cluster_stats['mean'].copy()
                mean_stats_original = self.transformer.inverse_transform(mean_stats)
                
                # Handle possible index changes
                if cluster_id in mean_stats_original.index:
                    stats = mean_stats_original.loc[cluster_id]
                else:
                    # Fall back to the original stats
                    stats = self.cluster_stats['mean'].loc[cluster_id]
                    print(f"Warning: Could not get inverse transformed stats for cluster {cluster_id}")
            except Exception as e:
                print(f"Warning: Error in inverse transformation for naming: {str(e)}")
                # Fall back to the original stats
                stats = self.cluster_stats['mean'].loc[cluster_id]
        else:
            stats = self.cluster_stats['mean'].loc[cluster_id]
        
        # Define the naming logic based on key features
        # Check if standard RFM features exist
        has_rfm = all(f in stats.index for f in ['recency_days', 'frequency', 'monetary'])
        has_satisfaction = 'avg_review_score' in stats.index
        
        # Start building the name
        name_parts = []
        
        if has_rfm:
            # Get recency thresholds from config
            recency_thresholds = self.segment_config['recency']
            # Recency classification
            if stats['recency_days'] < recency_thresholds['active']:
                name_parts.append("Active")
            elif stats['recency_days'] < recency_thresholds['recent']:
                name_parts.append("Recent")
            else:
                name_parts.append("Inactive")
                
            # Get frequency thresholds from config
            frequency_thresholds = self.segment_config['frequency']
            # Frequency classification
            if stats['frequency'] > frequency_thresholds['frequent']:
                name_parts.append("Frequent")
            elif stats['frequency'] > frequency_thresholds['returning']:
                name_parts.append("Returning")
            else:
                name_parts.append("One-time")
                
            # Get monetary threshold from config
            monetary_thresholds = self.segment_config['monetary']
            # Monetary classification
            high_value_threshold = monetary_thresholds['high_value']
            if high_value_threshold == 'mean':
                high_value_threshold = self.cluster_stats['mean']['monetary'].mean()
                
            if 'monetary' in stats and stats['monetary'] > high_value_threshold:
                name_parts.append("High-value")
            else:
                name_parts.append("Standard-value")
        
        # Add satisfaction if available
        if has_satisfaction:
            # Get satisfaction thresholds from config
            satisfaction_thresholds = self.segment_config['satisfaction']
            
            score = stats['avg_review_score']
            if score >= satisfaction_thresholds['very_satisfied']:
                name_parts.append("Very Satisfied")
            elif score >= satisfaction_thresholds['satisfied']:
                name_parts.append("Satisfied")
            elif score >= satisfaction_thresholds['neutral']:
                name_parts.append("Neutral")
            else:
                name_parts.append("Unsatisfied")
        
        # Combine parts (limit to 3 parts for brevity)
        if len(name_parts) > 3:
            name_parts = name_parts[:3]
            
        if not name_parts:
            return f"Cluster {cluster_id}"
            
        return " ".join(name_parts)
    
    def _generate_cluster_profiles(self):
        """Generate profiles for each cluster with descriptive names and key metrics."""
        self.cluster_profiles = {}
        
        # Handle case with no statistics - setup minimal profiles
        if 'mean' not in self.cluster_stats or len(self.cluster_stats['mean']) == 0:
            print("Warning: No cluster statistics available. Creating minimal profiles.")
            for cluster_id in sorted(self.clustered_data['cluster'].unique()):
                self.cluster_profiles[cluster_id] = {
                    'name': f"Cluster {cluster_id}",
                    'size': int(self.cluster_sizes.get(cluster_id, 0)),
                    'percentage': f"{self.cluster_sizes.get(cluster_id, 0) / sum(self.cluster_sizes):.1%}",
                    'key_metrics': {},
                    'categorical_metrics': {},
                    'insights': [f"Cluster {cluster_id} contains {self.cluster_sizes.get(cluster_id, 0)} samples"],
                    'original_mean': pd.Series(dtype='float64'),
                    'original_median': pd.Series(dtype='float64')
                }
            return
        
        for cluster_id in sorted(self.cluster_stats['mean'].index):
            # Get statistics for this cluster
            mean_stats = self.cluster_stats['mean'].loc[cluster_id]
            median_stats = self.cluster_stats['median'].loc[cluster_id]
            
            # Get original scale values if transformer is available
            if self.transformer is not None:
                try:
                    # Create a DataFrame with the values from this cluster
                    stats_df = pd.DataFrame([mean_stats, median_stats], 
                                          index=['mean', 'median'])
                    original_stats = self.transformer.inverse_transform(stats_df)
                    
                    # Check if index was preserved, if not handle differently
                    if 'mean' in original_stats.index:
                        original_mean = original_stats.loc['mean']
                        original_median = original_stats.loc['median']
                    else:
                        # If index was lost, assume the first row is mean and second is median
                        original_mean = original_stats.iloc[0]
                        original_median = original_stats.iloc[1]
                except Exception as e:
                    print(f"Warning: Error in inverse transformation: {str(e)}")
                    # Fall back to using untransformed values
                    original_mean = mean_stats
                    original_median = median_stats
            else:
                original_mean = mean_stats
                original_median = median_stats
            
            # Generate cluster name
            cluster_name = self._generate_cluster_profile_name(cluster_id)
            
            # Create profile with original scale metrics
            key_metrics = {}
            
            # Determine key metrics based on available features
            if 'recency_days' in original_mean:
                key_metrics['Days Since Last Purchase'] = f"{original_mean['recency_days']:.0f}"
            
            if 'frequency' in original_mean:
                key_metrics['Number of Orders'] = f"{original_mean['frequency']:.1f}"
            
            if 'monetary' in original_mean and 'frequency' in original_mean:
                avg_order = original_mean['monetary'] / max(1, original_mean['frequency'])
                key_metrics['Average Order Value'] = f"${avg_order:.2f}"
                key_metrics['Total Customer Value'] = f"${original_mean['monetary']:.2f}"
            
            if 'avg_review_score' in original_mean:
                key_metrics['Satisfaction Score'] = f"{original_mean['avg_review_score']:.1f}/5.0"
            
            if 'avg_delivery_time' in original_mean:
                key_metrics['Average Delivery Time'] = f"{original_mean['avg_delivery_time']:.1f} days"
            
            # Add categorical metrics if available
            cat_metrics = {}
            if 'categorical' in self.cluster_stats and cluster_id in self.cluster_stats['categorical']:
                for feature, stats in self.cluster_stats['categorical'][cluster_id].items():
                    cat_metrics[feature] = {
                        'most_common': stats['mode'],
                        'frequency': f"{stats['frequency']:.0%}"
                    }
            
            # Generate business insights
            insights = self._generate_insights(cluster_id, original_mean, original_median)
            
            # Store the profile
            self.cluster_profiles[cluster_id] = {
                'name': cluster_name,
                'size': int(self.cluster_sizes[cluster_id]),
                'percentage': f"{self.cluster_sizes[cluster_id] / sum(self.cluster_sizes):.1%}",
                'key_metrics': key_metrics,
                'categorical_metrics': cat_metrics,
                'insights': insights,
                'original_mean': original_mean,
                'original_median': original_median
            }
    
    def _generate_insights(self, cluster_id: int, mean_stats: pd.Series, median_stats: pd.Series) -> List[str]:
        """
        Generate business insights for a specific cluster.
        
        Parameters:
        -----------
        cluster_id : int
            Cluster identifier
        mean_stats : pd.Series
            Mean statistics for the cluster (original scale)
        median_stats : pd.Series
            Median statistics for the cluster (original scale)
            
        Returns:
        --------
        List[str]
            List of business insights for the cluster
        """
        insights = []
        
        # Global mean for comparison
        global_mean = self.cluster_stats['mean'].mean()
        if self.transformer is not None:
            try:
                global_mean_df = pd.DataFrame([global_mean], index=['global'])
                global_mean_transformed = self.transformer.inverse_transform(global_mean_df)
                
                # Handle possible index changes
                if 'global' in global_mean_transformed.index:
                    global_mean = global_mean_transformed.loc['global']
                else:
                    global_mean = global_mean_transformed.iloc[0]
            except Exception as e:
                print(f"Warning: Error in global mean inverse transformation: {str(e)}")
                # Continue with transformed global mean
    
        # Check if RFM features exist
        has_recency = 'recency_days' in mean_stats
        has_frequency = 'frequency' in mean_stats
        has_monetary = 'monetary' in mean_stats
        has_satisfaction = 'avg_review_score' in mean_stats
        
        # Get thresholds from config
        recency_thresholds = self.segment_config['recency']
        satisfaction_thresholds = self.segment_config['satisfaction']
        
        # Generate insights based on comparison with global average
        if has_recency:
            recency = mean_stats['recency_days']
            if recency < recency_thresholds['active']:
                insights.append("Recently active customers - ideal for upselling opportunities.")
            elif recency > recency_thresholds['recent']:
                insights.append("Inactive customers - consider reactivation campaigns.")
                
        if has_frequency and has_monetary:
            freq = mean_stats['frequency']
            monetary = mean_stats['monetary']
            global_freq = global_mean['frequency'] if 'frequency' in global_mean else 1
            global_monetary = global_mean['monetary'] if 'monetary' in global_mean else 0
            
            if freq > global_freq and monetary > global_monetary:
                insights.append("High-value loyal customers - prioritize retention strategies.")
            elif freq < global_freq and monetary > global_monetary:
                insights.append("Big spenders but infrequent - opportunity to increase purchase frequency.")
            elif freq > global_freq and monetary < global_monetary:
                insights.append("Frequent small purchases - potential for basket size expansion.")
                
        if has_satisfaction:
            score = mean_stats['avg_review_score']
            if score < satisfaction_thresholds['neutral']:
                insights.append("Below average satisfaction - investigate service/product issues.")
            elif score >= satisfaction_thresholds['very_satisfied']:
                insights.append("Highly satisfied customers - potential brand advocates.")
                
        # If we don't have enough insights, add a generic one
        if len(insights) == 0:
            insights.append(f"Cluster {cluster_id} represents {self.cluster_profiles[cluster_id]['percentage']} of customers.")
            
        return insights
    
    def create_cluster_size_plot(self) -> go.Figure:
        """
        Create a pie chart showing the distribution of clusters.
        
        Returns:
        --------
        plotly.graph_objects.Figure
            Pie chart of cluster sizes with descriptive names
        """
        # Prepare data
        labels = [f"Cluster {k}: {v['name']}" for k, v in self.cluster_profiles.items()]
        values = [v['size'] for k, v in self.cluster_profiles.items()]
        
        # Create color list that matches cluster ids
        colors = [self.cluster_colors[i % len(self.cluster_colors)] for i in self.cluster_profiles.keys()]
        
        # Create figure
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors),
            textinfo='percent+label',
            hoverinfo='label+value',
            hole=0.3
        )])
        
        fig.update_layout(
            title="Customer Segment Distribution",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            height=600,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
    
    def create_cluster_radar_comparison(self, 
                                      features: Optional[List[str]] = None,
                                      normalize: bool = True) -> go.Figure:
        """
        Create a radar chart comparing key metrics across clusters.
        
        Parameters:
        -----------
        features : List[str], optional
            Features to include in the radar chart (defaults to top 6 numeric features)
        normalize : bool
            Whether to normalize the values for better comparison
            
        Returns:
        --------
        plotly.graph_objects.Figure
            Radar chart comparing clusters
        """
        # If features not specified, use top features based on variance
        if features is None:
            # Calculate variance for each feature
            variances = self.data[self.numeric_features].var().sort_values(ascending=False)
            features = variances.index[:min(6, len(variances))].tolist()
        
        # Get data for radar chart
        radar_data = self.cluster_stats['mean'][features]
        
        # Normalize data if requested
        if normalize:
            radar_data = (radar_data - radar_data.min()) / (radar_data.max() - radar_data.min())
        
        # Create radar chart
        fig = go.Figure()
        
        # Add traces for each cluster
        for cluster_id in sorted(radar_data.index):
            values = radar_data.loc[cluster_id].tolist()
            # Close the loop for radar chart
            values.append(values[0])
            features_loop = features.copy()
            features_loop.append(features[0])
            
            cluster_name = self.cluster_profiles[cluster_id]['name']
            color = self.cluster_colors[cluster_id % len(self.cluster_colors)]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=features_loop,
                fill='toself',
                name=f"Cluster {cluster_id}: {cluster_name}",
                line=dict(color=color),
                fillcolor=color.replace('rgb', 'rgba').replace(')', ', 0.2)')
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1] if normalize else None
                )
            ),
            title="Cluster Feature Comparison",
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_feature_distribution_plot(self, feature: str) -> go.Figure:
        """
        Create a boxplot showing the distribution of a feature across clusters.
        
        Parameters:
        -----------
        feature : str
            Feature to visualize
            
        Returns:
        --------
        plotly.graph_objects.Figure
            Boxplot of feature distributions by cluster
        """
        # Check if feature exists
        if feature not in self.data.columns:
            raise ValueError(f"Feature '{feature}' not found in data")
        
        # Create figure
        fig = go.Figure()
        
        # Add trace for each cluster
        for cluster_id in sorted(self.clustered_data['cluster'].unique()):
            cluster_data = self.clustered_data[self.clustered_data['cluster'] == cluster_id]
            
            cluster_name = self.cluster_profiles[cluster_id]['name']
            color = self.cluster_colors[cluster_id % len(self.cluster_colors)]
            
            fig.add_trace(go.Box(
                y=cluster_data[feature],
                name=f"Cluster {cluster_id}",
                marker_color=color,
                boxmean=True,  # Shows the mean as a dashed line
                hovertext=[f"{cluster_name}"] * len(cluster_data),
                hoverinfo='y+text'
            ))
        
        # Get feature in original scale name for better labels
        feature_name = feature
        if self.transformer is not None and hasattr(self.transformer, 'get_feature_names'):
            feature_name = self.transformer.get_feature_names().get(feature, feature)
        
        fig.update_layout(
            title=f"Distribution of {feature_name} Across Clusters",
            yaxis_title=feature_name,
            xaxis_title="Cluster",
            boxmode='group',
            height=500
        )
        
        return fig
    
    def create_cluster_metrics_table(self) -> go.Figure:
        """
        Create a table with key metrics for all clusters.
        
        Returns:
        --------
        plotly.graph_objects.Figure
            Table with key metrics
        """
        # Prepare table data
        table_data = []
        
        # Column headers
        headers = ['Cluster', 'Name', 'Size', 'Percentage']
        
        # Find common metrics across all clusters
        common_metrics = set()
        for profile in self.cluster_profiles.values():
            common_metrics.update(profile['key_metrics'].keys())
        common_metrics = sorted(list(common_metrics))
        
        # Add metric headers
        headers.extend(common_metrics)
        
        # Add insight header
        headers.append('Key Insight')
        
        # Add data for each cluster
        for cluster_id, profile in sorted(self.cluster_profiles.items()):
            row = [
                f"Cluster {cluster_id}",
                profile['name'],
                profile['size'],
                profile['percentage']
            ]
            
            # Add metrics
            for metric in common_metrics:
                row.append(profile['key_metrics'].get(metric, '-'))
            
            # Add first insight
            if profile['insights']:
                row.append(profile['insights'][0])
            else:
                row.append('-')
            
            table_data.append(row)
        
        # Create figure
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=headers,
                fill_color='paleturquoise',
                align='left',
                font=dict(size=12)
            ),
            cells=dict(
                values=list(map(list, zip(*table_data))),  # Transpose data for table
                fill_color='lavender',
                align='left'
            )
        )])
        
        fig.update_layout(
            title="Customer Segment Metrics",
            height=len(table_data) * 40 + 100,  # Adjust height based on number of rows
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
    
    def create_segment_action_table(self) -> go.Figure:
        """
        Create a table with recommended actions for each cluster.
        
        Returns:
        --------
        plotly.graph_objects.Figure
            Table with recommended actions
        """
        # Generate recommended actions based on cluster characteristics
        actions = {}
        
        for cluster_id, profile in self.cluster_profiles.items():
            mean_stats = profile['original_mean']
            cluster_actions = []
            
            # RFM-based actions
            if 'recency_days' in mean_stats and 'frequency' in mean_stats and 'monetary' in mean_stats:
                recency = mean_stats['recency_days']
                frequency = mean_stats['frequency']
                monetary = mean_stats['monetary']
                
                # High value + Recent
                if recency < 30 and monetary > 150:
                    cluster_actions.append("Upsell premium products/services")
                    cluster_actions.append("Offer loyalty rewards")
                
                # High value + Not recent
                elif recency > 90 and monetary > 150:
                    cluster_actions.append("Reactivation campaign with premium offer")
                    cluster_actions.append("Personal outreach to understand churn reasons")
                
                # High frequency
                if frequency > 2:
                    cluster_actions.append("Cross-sell complementary products")
                    cluster_actions.append("Subscription or bundle offers")
                
                # Low monetary
                if monetary < 100:
                    cluster_actions.append("Product education to show value")
                    cluster_actions.append("Limited-time discounts on premium alternatives")
            
            # Satisfaction-based actions
            if 'avg_review_score' in mean_stats:
                score = mean_stats['avg_review_score']
                
                if score < 3.5:
                    cluster_actions.append("Service recovery outreach")
                    cluster_actions.append("Gather feedback on improvement areas")
                elif score > 4.5:
                    cluster_actions.append("Request referrals and testimonials")
                    cluster_actions.append("Early access to new products/features")
            
            # Ensure we have at least 3 actions
            if len(cluster_actions) < 3:
                cluster_actions.append("Monitor segment performance monthly")
                
            # Limit to top 3 actions
            actions[cluster_id] = cluster_actions[:3]
        
        # Prepare table data
        headers = ['Cluster', 'Segment Name', 'Key Characteristics', 'Recommended Actions']
        table_data = []
        
        for cluster_id, profile in sorted(self.cluster_profiles.items()):
            # Extract key characteristics from metrics
            characteristics = []
            if 'Days Since Last Purchase' in profile['key_metrics']:
                recency = float(profile['key_metrics']['Days Since Last Purchase'])
                if recency < 30:
                    characteristics.append("Recently active")
                elif recency > 90:
                    characteristics.append("Inactive")
            
            if 'Average Order Value' in profile['key_metrics'] and 'Number of Orders' in profile['key_metrics']:
                aov = profile['key_metrics']['Average Order Value'].replace('$', '')
                frequency = profile['key_metrics']['Number of Orders']
                if float(aov) > 150:
                    characteristics.append("High spender")
                if float(frequency) > 2:
                    characteristics.append("Frequent buyer")
            
            if 'Satisfaction Score' in profile['key_metrics']:
                score = profile['key_metrics']['Satisfaction Score'].split('/')[0]
                if float(score) > 4.5:
                    characteristics.append("Very satisfied")
                elif float(score) < 3.5:
                    characteristics.append("Unsatisfied")
            
            # Limit to top 3 characteristics
            characteristics = characteristics[:3]
            if not characteristics:
                characteristics = [f"{profile['percentage']} of customer base"]
            
            # Create row
            row = [
                f"Cluster {cluster_id}",
                profile['name'],
                ", ".join(characteristics),
                "\n".join([f"• {action}" for action in actions[cluster_id]])
            ]
            table_data.append(row)
        
        # Create figure
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=headers,
                fill_color='lightblue',
                align='left',
                font=dict(size=14)
            ),
            cells=dict(
                values=list(map(list, zip(*table_data))),  # Transpose data for table
                fill_color='lavender',
                align='left',
                height=40  # Increase cell height for multiline text
            )
        )])
        
        fig.update_layout(
            title="Recommended Actions by Customer Segment",
            height=len(table_data) * 60 + 100,  # Adjust height for multiline text
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
    
    def create_dashboard(self) -> Dict[str, go.Figure]:
        """
        Create a complete dashboard with multiple visualizations.
        
        Returns:
        --------
        Dict[str, plotly.graph_objects.Figure]
            Dictionary of named dashboard components
        """
        dashboard = {}
        
        try:
            # Add cluster size plot
            dashboard['cluster_sizes'] = self.create_cluster_size_plot()
            
            # Add radar comparison of key metrics if we have enough features
            if len(self.numeric_features) >= 3:
                try:
                    dashboard['feature_comparison'] = self.create_cluster_radar_comparison()
                except Exception as e:
                    print(f"Warning: Could not create radar comparison: {str(e)}")
                    # Add a simple error figure instead
                    fig = go.Figure()
                    fig.add_annotation(text=f"Could not create radar chart: {str(e)}",
                                      xref="paper", yref="paper",
                                      x=0.5, y=0.5, showarrow=False)
                    dashboard['feature_comparison'] = fig
            
            # Add metrics table
            try:
                dashboard['metrics_table'] = self.create_cluster_metrics_table()
            except Exception as e:
                print(f"Warning: Could not create metrics table: {str(e)}")
                # Add a simple error figure
                fig = go.Figure()
                fig.add_annotation(text=f"Could not create metrics table: {str(e)}",
                                  xref="paper", yref="paper",
                                  x=0.5, y=0.5, showarrow=False)
                dashboard['metrics_table'] = fig
            
            # Add action recommendations
            try:
                dashboard['action_table'] = self.create_segment_action_table()
            except Exception as e:
                print(f"Warning: Could not create action table: {str(e)}")
                fig = go.Figure()
                fig.add_annotation(text=f"Could not create action table: {str(e)}",
                                  xref="paper", yref="paper",
                                  x=0.5, y=0.5, showarrow=False)
                dashboard['action_table'] = fig
            
            # Add feature distributions for top features by variance
            try:
                if self.numeric_features:
                    # Get top features by variance
                    feature_vars = self.data[self.numeric_features].var().sort_values(ascending=False)
                    key_features = feature_vars.index[:min(3, len(feature_vars))].tolist()
                    
                    # Create distribution plots
                    for feature in key_features:
                        try:
                            dashboard[f"feature_distribution_{feature}"] = self.create_feature_distribution_plot(feature)
                        except Exception as e:
                            print(f"Warning: Could not create distribution for {feature}: {str(e)}")
            except Exception as e:
                print(f"Warning: Could not create feature distribution plots: {str(e)}")
    
        except Exception as e:
            print(f"Error creating dashboard component: {str(e)}")
            # Return a minimal dashboard with an error message
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating dashboard: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="red")
            )
            dashboard['error'] = fig
        
        return dashboard
def create_sales_dashboard(cluster_labels, feature_data, transformer=None, segment_config=None):
    """
    Create a sales dashboard from cluster labels and feature data.
    
    Parameters:
    -----------
    cluster_labels : numpy.ndarray
        Array of cluster assignments for each customer
    feature_data : pandas.DataFrame
        DataFrame containing customer features (transformed or original)
    transformer : GenericFeatureTransformer, optional
        Transformer to convert features to original scale
    segment_config : Dict, optional
        Dictionary containing threshold configurations for segment naming
    
    Returns:
    --------
    Dict[str, plotly.graph_objects.Figure]
        Dictionary of dashboard components
    """
    dashboard = ClusterDashboard(cluster_labels, feature_data, transformer, segment_config)
    return dashboard.create_dashboard()
