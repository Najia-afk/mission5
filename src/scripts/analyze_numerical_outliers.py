import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def analyze_outliers_with_multiple_methods(df):
    """
    Analyze outliers in numerical data using different detection methods with interactive dropdown
    
    Parameters:
        df (pd.DataFrame): Input dataframe with numerical columns
        
    Returns:
        tuple: (all_summaries, all_cleaned_dfs) - Dictionary of outlier summary DataFrames 
               and dictionary of cleaned DataFrames for each method
    """
    # Define outlier detection methods
    methods = {
        "Z-score (±3)": {
            "function": lambda col: (
                col.mean() - 3 * col.std(),
                col.mean() + 3 * col.std()
            ),
            "description": "Mean ± 3×Std. Dev. (common statistical approach)"
        },
        "IQR 1.5": {
            "function": lambda col: (
                col.quantile(0.25) - 1.5 * (col.quantile(0.75) - col.quantile(0.25)),
                col.quantile(0.75) + 1.5 * (col.quantile(0.75) - col.quantile(0.25))
            ),
            "description": "Standard method: Q1-1.5×IQR to Q3+1.5×IQR"
        },
        "Z-score (±2)": {
            "function": lambda col: (
                col.mean() - 2 * col.std(),
                col.mean() + 2 * col.std()
            ),
            "description": "Mean ± 2×Std. Dev. (stricter approach)"
        },
        "Quantile 1-99": {
            "function": lambda col: (col.quantile(0.01), col.quantile(0.99)),
            "description": "1st to 99th percentile range"
        },
        "Quantile 5-95": {
            "function": lambda col: (col.quantile(0.05), col.quantile(0.95)),
            "description": "5th to 95th percentile range" 
        },
        "Full Range": {
            "function": lambda col: (col.min(), col.max()),
            "description": "No outlier filtering (min to max)"
        }
    }
    
    # Get numeric columns only
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Create dictionaries to store results for each method
    all_summaries = {}
    all_cleaned_dfs = {}
    all_outlier_info = {}
    all_stats_info = {}
    
    # Process each method
    for method_name, method_info in methods.items():
        # Create a clean copy for this method
        df_clean = df.copy()
        
        # Dictionaries to store outlier information for this method
        outlier_info = {}
        stats_info = {}
        
        # Apply this method to each numeric column
        for col in numeric_cols:
            # Skip if no data in column
            if df[col].count() == 0:
                continue
                
            # Get bounds using this method
            lower_bound, upper_bound = method_info["function"](df[col])
            
            # Identify outliers based on bounds
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound))
            
            # Calculate statistics
            mean = df[col].mean()
            median = df[col].median()
            std_dev = df[col].std()
            clean_mean = df[col][~outliers].mean() if (~outliers).any() else np.nan
            clean_median = df[col][~outliers].median() if (~outliers).any() else np.nan
            clean_std_dev = df[col][~outliers].std() if (~outliers).any() else np.nan
            
            # Calculate percentage change in mean due to outliers
            if pd.notna(clean_mean) and clean_mean != 0:  
                mean_percent_change = ((mean - clean_mean) / clean_mean) * 100
            else:
                mean_percent_change = 0
            
            # Store statistics
            stats_info[col] = {
                'mean': mean,
                'median': median, 
                'std_dev': std_dev,
                'clean_mean': clean_mean,
                'clean_median': clean_median,
                'clean_std_dev': clean_std_dev,
                'mean_percent_change': mean_percent_change,
                'skewness': df[col].skew()
            }
            
            # Store outlier information
            outlier_info[col] = {
                'outlier_count': outliers.sum(),
                'outlier_percentage': outliers.sum() / df[col].count() * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'extreme_min': df[col][outliers & (df[col] < lower_bound)].min() if any(outliers & (df[col] < lower_bound)) else None,
                'extreme_max': df[col][outliers & (df[col] > upper_bound)].max() if any(outliers & (df[col] > upper_bound)) else None,
                'num_below_lower': (outliers & (df[col] < lower_bound)).sum(),
                'num_above_upper': (outliers & (df[col] > upper_bound)).sum()
            }
            
            # Set outliers to NaN in clean dataframe (you can change to capping if preferred)
            df_clean.loc[outliers, col] = np.nan
        
        # Create a summary DataFrame for this method
        summary_df = pd.DataFrame({
            'Variable': [col for col in outlier_info.keys()],
            'Outlier Count': [info['outlier_count'] for info in outlier_info.values()],
            'Outlier %': [f"{info['outlier_percentage']:.2f}%" for info in outlier_info.values()],
            'Below Min': [info['num_below_lower'] for info in outlier_info.values()],
            'Above Max': [info['num_above_upper'] for info in outlier_info.values()],
            'Min Limit': [info['lower_bound'] for info in outlier_info.values()],
            'Max Limit': [info['upper_bound'] for info in outlier_info.values()],
            'Extreme Min': [info['extreme_min'] for info in outlier_info.values()],
            'Extreme Max': [info['extreme_max'] for info in outlier_info.values()],
            'Mean (with outliers)': [stats_info[col]['mean'] for col in outlier_info.keys()],
            'Mean (w/o outliers)': [stats_info[col]['clean_mean'] for col in outlier_info.keys()],
            'Mean % Change': [stats_info[col]['mean_percent_change'] for col in outlier_info.keys()],
            'Skewness': [stats_info[col]['skewness'] for col in outlier_info.keys()]
        })
        
        # Sort by outlier count
        summary_df = summary_df.sort_values(by='Outlier Count', ascending=False)
        
        # Store results for this method
        all_summaries[method_name] = summary_df
        all_cleaned_dfs[method_name] = df_clean
        all_outlier_info[method_name] = outlier_info
        all_stats_info[method_name] = stats_info
    
    # Create interactive visualization with method dropdown
    fig = create_outlier_visualization(all_summaries, all_outlier_info, all_stats_info, methods)
    fig.show()
    
    return all_summaries, all_cleaned_dfs

def create_outlier_visualization(all_summaries, all_outlier_info, all_stats_info, methods):
    """Create an interactive outlier visualization with method dropdown"""
    
    # Get the list of methods and default to first method
    method_names = list(all_summaries.keys())
    default_method = method_names[0]
    
    # Create the figure with subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Outlier Counts by Variable", "Impact on Mean Values (% Change)"),
        vertical_spacing=0.3,
        specs=[[{"type": "bar"}], [{"type": "bar"}]]
    )
    
    # Add traces for each method (initially visible only for default method)
    for method_name in method_names:
        summary_df = all_summaries[method_name]
        
        # Determine visibility
        visible = (method_name == default_method)
        
        # Add bar chart for outlier counts
        fig.add_trace(
            go.Bar(
                x=summary_df['Variable'],
                y=summary_df['Outlier Count'],
                name='Outlier Count',
                marker_color='red',
                text=summary_df['Outlier %'],
                hovertemplate='%{x}<br>Outliers: %{y}<br>(%{text})<extra></extra>',
                visible=visible
            ),
            row=1, col=1
        )
        
        # Add bar chart for mean percent change
        fig.add_trace(
            go.Bar(
                x=summary_df['Variable'],
                y=summary_df['Mean % Change'],
                name='Mean % Change',
                marker_color='purple',
                text=[f"{x:.1f}%" for x in summary_df['Mean % Change']],
                textposition='auto',
                hovertemplate='%{x}<br>Mean % Change: %{text}<extra></extra>',
                visible=visible
            ),
            row=2, col=1
        )
    
    # Create dropdown buttons
    dropdown_buttons = []
    for i, method_name in enumerate(method_names):
        visibility = [method == method_name for method in method_names for _ in range(2)]  # 2 traces per method
        
        dropdown_buttons.append(
            dict(
                label=f"{method_name} - {methods[method_name]['description']}",
                method="update",
                args=[
                    {"visible": visibility},
                    {"title": f"Outlier Analysis using {method_name} Method"}
                ]
            )
        )
    
    # Update layout with dropdown menu
    fig.update_layout(
        title={
            'text': f"Outlier Analysis using {default_method} Method",
            'x': 0.5,
            'xanchor': 'center'
        },
        height=800,
        showlegend=False,  # Hide legend since we use the dropdown instead
        updatemenus=[
            dict(
                active=0,
                buttons=dropdown_buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.7,
                xanchor="left",
                y=1.15,
                yanchor="top"
            )
        ],
        margin=dict(r=200)  # Add right margin for annotation
    )
    
    fig.update_xaxes(tickangle=45)
    fig.update_yaxes(title_text="Outlier Count", row=1, col=1)
    fig.update_yaxes(title_text="Mean % Change", row=2, col=1)
    
    return fig

# Add a function to show detailed statistics for a specific variable with all methods
def compare_variable_outlier_methods(df, variable_name):
    """
    Create detailed visualization comparing outlier detection methods for a specific variable
    
    Parameters:
        df (pd.DataFrame): Input dataframe
        variable_name (str): Name of the variable to analyze
    """
    if variable_name not in df.columns:
        print(f"Variable '{variable_name}' not found in dataframe.")
        return
    
    # Define outlier detection methods
    methods = {
        "Z-score (±3)": {
            "function": lambda col: (
                col.mean() - 3 * col.std(),
                col.mean() + 3 * col.std()
            ),
            "color": "rgba(148, 103, 189, 0.7)"
        },
        "IQR 1.5": {
            "function": lambda col: (
                col.quantile(0.25) - 1.5 * (col.quantile(0.75) - col.quantile(0.25)),
                col.quantile(0.75) + 1.5 * (col.quantile(0.75) - col.quantile(0.25))
            ),
            "color": "rgba(31, 119, 180, 0.7)"
        },
        "Z-score (±2)": {
            "function": lambda col: (
                col.mean() - 2 * col.std(),
                col.mean() + 2 * col.std()
            ),
            "color": "rgba(140, 86, 75, 0.7)"
        },
        "Quantile 1-99": {
            "function": lambda col: (col.quantile(0.01), col.quantile(0.99)),
            "color": "rgba(255, 127, 14, 0.7)"
        },
        "Quantile 5-95": {
            "function": lambda col: (col.quantile(0.05), col.quantile(0.95)),
            "color": "rgba(44, 160, 44, 0.7)"
        },
        "Full Range": {
            "function": lambda col: (col.min(), col.max()),
            "color": "rgba(214, 39, 40, 0.7)"
        }
    }
    
    # Extract the variable data
    data = df[variable_name].dropna()
    
    # Calculate statistics for each method
    method_stats = {}
    for method_name, method_info in methods.items():
        lower_bound, upper_bound = method_info["function"](data)
        
        # Identify outliers
        outliers = ((data < lower_bound) | (data > upper_bound))
        clean_data = data[~outliers]
        
        # Calculate statistics
        method_stats[method_name] = {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "outlier_count": outliers.sum(),
            "outlier_percentage": (outliers.sum() / len(data)) * 100,
            "mean_with_outliers": data.mean(),
            "mean_without_outliers": clean_data.mean() if len(clean_data) > 0 else np.nan,
            "median_with_outliers": data.median(),
            "median_without_outliers": clean_data.median() if len(clean_data) > 0 else np.nan,
            "std_with_outliers": data.std(),
            "std_without_outliers": clean_data.std() if len(clean_data) > 0 else np.nan,
            "color": method_info["color"]
        }
    
    # Create visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f"Distribution of {variable_name} with Different Bounds",
            "Outlier Counts by Method",
            "Impact on Mean",
            "Impact on Standard Deviation"
        ),
        specs=[
            [{"type": "histogram"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "bar"}]
        ],
        vertical_spacing=0.2,
        horizontal_spacing=0.1
    )
    
    # 1. Distribution histogram with bounds
    fig.add_trace(
        go.Histogram(
            x=data,
            nbinsx=30,
            name="Distribution",
            marker_color="rgba(100, 100, 100, 0.5)",
            opacity=0.7
        ),
        row=1, col=1
    )
    
    # Add vertical lines for bounds
    for method_name, stats in method_stats.items():
        # Lower bound
        fig.add_trace(
            go.Scatter(
                x=[stats["lower_bound"], stats["lower_bound"]],
                y=[0, data.value_counts().max()],
                mode="lines",
                name=f"{method_name} Lower",
                line=dict(color=stats["color"], width=2, dash="dash")
            ),
            row=1, col=1
        )
        
        # Upper bound
        fig.add_trace(
            go.Scatter(
                x=[stats["upper_bound"], stats["upper_bound"]],
                y=[0, data.value_counts().max()],
                mode="lines",
                name=f"{method_name} Upper",
                line=dict(color=stats["color"], width=2, dash="dash")
            ),
            row=1, col=1
        )
    
    # 2. Outlier counts bar chart
    fig.add_trace(
        go.Bar(
            x=list(methods.keys()),
            y=[stats["outlier_count"] for stats in method_stats.values()],
            marker_color=[stats["color"] for stats in method_stats.values()],
            text=[f"{stats['outlier_percentage']:.1f}%" for stats in method_stats.values()],
            textposition="auto",
            hovertemplate="Method: %{x}<br>Outliers: %{y}<br>(%{text})<extra></extra>"
        ),
        row=1, col=2
    )
    
    # 3. Mean comparison
    means_with = [stats["mean_with_outliers"] for stats in method_stats.values()]
    means_without = [stats["mean_without_outliers"] for stats in method_stats.values()]
    
    fig.add_trace(
        go.Bar(
            x=list(methods.keys()),
            y=means_with,
            name="With Outliers",
            marker_color="rgba(31, 119, 180, 0.7)",
            text=[f"{m:.2f}" for m in means_with],
            textposition="auto",
            hovertemplate="Method: %{x}<br>Mean with outliers: %{text}<extra></extra>"
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=list(methods.keys()),
            y=means_without,
            name="Without Outliers",
            marker_color="rgba(255, 127, 14, 0.7)",
            text=[f"{m:.2f}" for m in means_without],
            textposition="auto",
            hovertemplate="Method: %{x}<br>Mean without outliers: %{text}<extra></extra>"
        ),
        row=2, col=1
    )
    
    # 4. Standard deviation comparison
    stds_with = [stats["std_with_outliers"] for stats in method_stats.values()]
    stds_without = [stats["std_without_outliers"] for stats in method_stats.values()]
    
    fig.add_trace(
        go.Bar(
            x=list(methods.keys()),
            y=stds_with,
            name="With Outliers",
            marker_color="rgba(31, 119, 180, 0.7)",
            text=[f"{s:.2f}" for s in stds_with],
            textposition="auto",
            hovertemplate="Method: %{x}<br>StdDev with outliers: %{text}<extra></extra>"
        ),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Bar(
            x=list(methods.keys()),
            y=stds_without,
            name="Without Outliers",
            marker_color="rgba(255, 127, 14, 0.7)",
            text=[f"{s:.2f}" for s in stds_without],
            textposition="auto",
            hovertemplate="Method: %{x}<br>StdDev without outliers: %{text}<extra></extra>"
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': f"Detailed Outlier Analysis for {variable_name}",
            'x': 0.5,
            'xanchor': 'center'
        },
        height=800,
        width=1200,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5
        ),
        barmode="group",
        annotations=[
            dict(
                x=1.05,
                y=0.25,
                xref="paper",
                yref="paper",
                xanchor="left",
                yanchor="middle",
                text="<b>Method Comparison</b><br><br>" + 
                     "• IQR 1.5: Interquartile range<br>" +
                     "• Quantile 1-99: 1% to 99%<br>" +
                     "• Quantile 5-95: 5% to 95%<br>" +
                     "• Full Range: No outlier filtering",
                showarrow=False,
                align="left",
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="gray",
                borderwidth=1,
                borderpad=10,
                opacity=0.9
            )
        ],
        margin=dict(r=200)
    )
    
    fig.update_xaxes(title_text=f"{variable_name} Value", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_xaxes(title_text="Method", row=1, col=2)
    fig.update_yaxes(title_text="Outlier Count", row=1, col=2)
    fig.update_xaxes(title_text="Method", row=2, col=1)
    fig.update_yaxes(title_text="Mean Value", row=2, col=1)
    fig.update_xaxes(title_text="Method", row=2, col=2)
    fig.update_yaxes(title_text="Standard Deviation", row=2, col=2)
    
    fig.show()
    
    # Return detailed statistics
    method_stats_df = pd.DataFrame({
        "Method": list(methods.keys()),
        "Lower Bound": [stats["lower_bound"] for stats in method_stats.values()],
        "Upper Bound": [stats["upper_bound"] for stats in method_stats.values()],
        "Outlier Count": [stats["outlier_count"] for stats in method_stats.values()],
        "Outlier %": [f"{stats['outlier_percentage']:.2f}%" for stats in method_stats.values()],
        "Mean (with outliers)": [stats["mean_with_outliers"] for stats in method_stats.values()],
        "Mean (without outliers)": [stats["mean_without_outliers"] for stats in method_stats.values()],
        "Mean % Change": [(stats["mean_with_outliers"] - stats["mean_without_outliers"]) / 
                         stats["mean_without_outliers"] * 100 if pd.notna(stats["mean_without_outliers"]) 
                         else np.nan for stats in method_stats.values()],
        "StdDev (with outliers)": [stats["std_with_outliers"] for stats in method_stats.values()],
        "StdDev (without outliers)": [stats["std_without_outliers"] for stats in method_stats.values()],
        "StdDev % Change": [(stats["std_with_outliers"] - stats["std_without_outliers"]) / 
                          stats["std_without_outliers"] * 100 if pd.notna(stats["std_without_outliers"]) 
                          else np.nan for stats in method_stats.values()]
    })
    
    print(f"Detailed statistics for {variable_name}:")
    display(method_stats_df)
    
    return method_stats_df