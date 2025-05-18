import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def create_quantile_distribution_plots(df, numeric_cols=None, target='TotalEnergy(kBtu)', n_quantiles=5):
    """
    Create interactive box plots showing distribution of numeric variables across target quantiles.
    
    Args:
        df: DataFrame containing the data
        numeric_cols: List of numeric columns to analyze (if None, all numeric columns are used)
        target: Column name to create quantiles from (e.g., 'TotalEnergy(kBtu)')
        n_quantiles: Number of quantiles to create (5 or 10)
        
    Returns:
        A plotly figure object
    """
    # Make a copy to avoid modifying the original
    df_temp = df.copy()
    
    # Filter out rows with missing target values
    df_temp = df_temp.dropna(subset=[target])
    
    # Create quantiles of the target variable
    df_temp['target_quantile'] = pd.qcut(df_temp[target], q=n_quantiles, labels=False)
    
    # Create quantile labels for display
    quantile_labels = []
    for q in range(n_quantiles):
        subset = df_temp[df_temp['target_quantile'] == q]
        if not subset.empty:
            min_val = subset[target].min()
            max_val = subset[target].max()
            quantile_labels.append(f'Q{q+1}: {min_val:.1f}-{max_val:.1f}')
    
    # Identify numeric columns if not specified
    if numeric_cols is None:
        numeric_cols = df_temp.select_dtypes(include=['number']).columns.tolist()
        # Remove target from the numeric columns
        if target in numeric_cols:
            numeric_cols.remove(target)
    
    # Create a single figure with dropdown menu
    fig = go.Figure()
    
    # Define quantile colors (similar to grade colors)
    quantile_colors = {
        0: '#038141',  # dark green
        1: '#85bb2f',  # light green
        2: '#fecb02',  # yellow
        3: '#ee8100',  # orange
        4: '#e63e11'   # red
    }
    
    # Add more colors if needed for more quantiles
    if n_quantiles > 5:
        more_colors = ['#4B0082', '#0000CD', '#1E90FF', '#20B2AA', '#9467BD']
        for i in range(5, min(n_quantiles, 10)):
            quantile_colors[i] = more_colors[i-5]
    
    # Add box plots for the first variable to make them initially visible
    if len(numeric_cols) > 0:
        first_var = numeric_cols[0]
        for q in range(n_quantiles):
            subset = df_temp[df_temp['target_quantile'] == q]
            
            # Skip quantiles with too few data points
            if len(subset) < 5:
                continue
                
            fig.add_trace(go.Box(
                y=subset[first_var],
                name=quantile_labels[q],
                boxmean=True,
                marker_color=quantile_colors.get(q, '#808080'),
                visible=True
            ))
    
    # Add all other variables as hidden traces
    for col in numeric_cols[1:]:
        for q in range(n_quantiles):
            subset = df_temp[df_temp['target_quantile'] == q]
            
            # Skip quantiles with too few data points
            if len(subset) < 5:
                continue
                
            fig.add_trace(go.Box(
                y=subset[col],
                name=quantile_labels[q],
                boxmean=True,
                marker_color=quantile_colors.get(q, '#808080'),
                visible=False
            ))
    
    # Create dropdown menu
    buttons = []
    for i, col in enumerate(numeric_cols):
        # Calculate the number of quantiles with enough data
        quantile_count = len([q for q in range(n_quantiles) 
                            if len(df_temp[df_temp['target_quantile'] == q]) >= 5])
        
        visibility = [False] * len(fig.data)
        for j in range(quantile_count):
            visibility[i * quantile_count + j] = True
            
        buttons.append(dict(
            label=format_column_name(col),
            method='update',
            args=[{'visible': visibility}, 
                  {'title': f'Distribution of {format_column_name(col)} by {target} Quantiles',
                   'yaxis': {'title': format_column_name(col)}}]
        ))
    
    # Update layout
    fig.update_layout(
        title=f'Distribution of {format_column_name(numeric_cols[0])} by {target} Quantiles',
        yaxis_title=format_column_name(numeric_cols[0]),
        xaxis_title=f'{target} Quantiles',
        updatemenus=[{
            'buttons': buttons,
            'direction': 'down',
            'showactive': True,
            'x': 0.75,
            'y': 1.10,
        },
        {
            'buttons': [
                {
                    'method': 'restyle',
                    'label': 'Box Plot',
                    'args': [{'type': 'box'}],
                },
                {
                    'method': 'restyle',
                    'label': 'Violin Plot',
                    'args': [{'type': 'violin', 'points': False, 'box': True, 'meanline': True}],
                }
            ],
            'direction': 'right',
            'x': 0.55,
            'y': 1.10,
            'showactive': True
        }],
        boxmode='group',
        height=600,
        legend_title_text=f'{target} Quantiles',
        margin=dict(l=40, r=150, t=80, b=40)
    )
    
    
    return fig

def format_column_name(col_name):
    """Format column name for display in the visualization"""
    # Replace underscores and hyphens with spaces
    formatted = col_name.replace('_', ' ').replace('-', ' ')

    
    # Capitalize words
    formatted = ' '.join(word.capitalize() for word in formatted.split())
    
    return formatted