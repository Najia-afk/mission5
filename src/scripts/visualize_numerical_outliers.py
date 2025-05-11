import plotly.graph_objects as go
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
from scipy import stats

def create_interactive_outlier_visualization(df, use_log_scale=False):
    """
    Create an interactive visualization to explore outliers in numeric columns
    
    Parameters:
        df (pd.DataFrame): Input dataframe (assumed to be already cleaned from outliers if df_already_clean=True)
        df_already_clean (bool): Whether the input dataframe is already cleaned from outliers (default=True)
        use_log_scale (bool): Use logarithmic scale for highly skewed data (default=True)
        
    Returns:
        tuple: (summary_df, df_clean) - Outlier summary DataFrame and cleaned DataFrame
    """
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if not numeric_cols:
        print("No numeric columns found in the dataframe")
        return
    
    # Create a copy of the dataframe for visualization
    df_clean = df.copy()
    
    # Create dictionary to store stats info
    stats_info = {}
    
    # Calculate statistics for each numeric column
    for col in numeric_cols:
        if df[col].count() == 0:
            continue
            
        # Check if the column has all positive values for log transform
        can_use_log = (df[col].min() > 0) if use_log_scale else False
        
        # Calculate statistics
        mean = df[col].mean()
        median = df[col].median()
        std_dev = df[col].std()
        
        # Store statistics
        stats_info[col] = {
            'mean': mean,
            'median': median, 
            'std_dev': std_dev,
            'clean_mean': mean,  # Same as mean since data is already clean
            'clean_median': median,  # Same as median since data is already clean
            'clean_std_dev': std_dev,  # Same as std_dev since data is already clean
            'can_use_log': can_use_log,
            'skewness': df[col].skew()
        }
    
    # Create a table with summary
    summary_df = pd.DataFrame({
        'Column': [col for col in stats_info.keys()],
        'Skewness': [stats_info[col]['skewness'] for col in stats_info.keys()],
        'Mean': [stats_info[col]['mean'] for col in stats_info.keys()],
        'Median': [stats_info[col]['median'] for col in stats_info.keys()],
        'StdDev': [stats_info[col]['std_dev'] for col in stats_info.keys()]
    })
    
    print("Data Summary:")
    display(summary_df)
    
    # Create the interactive figure - side by side layout
    fig = make_subplots(
        rows=1, 
        cols=2,
        subplot_titles=("Box Plot", "Distribution"),
        horizontal_spacing=0.1,
        specs=[[{"type": "box"}, {"type": "histogram"}]]
    )
    
    # Initialize with the first numeric column
    first_col = numeric_cols[0]
    
    # Create visualizations for each column
    for i, col in enumerate(numeric_cols):
        # Only calculate for non-empty columns
        if df[col].count() == 0:
            continue
            
        if i == 0:
            # For the first column, add to the plot - improved box plot
            data = df[col].dropna()
            
            # Pre-compute box plot stats for better rendering
            q1 = np.percentile(data, 25)
            median = np.percentile(data, 50)
            q3 = np.percentile(data, 75)
            iqr = q3 - q1
            whisker_min = max(np.min(data), q1 - 1.5 * iqr)
            whisker_max = min(np.max(data), q3 + 1.5 * iqr)
            
            fig.add_trace(
                go.Box(
                    name=col,
                    marker_color='red',
                    visible=True,
                    y=data,
                    boxmean=True,  # Show mean as a dashed line
                    notched=False,  # Don't use notched boxes
                    boxpoints='outliers',  # Only show outlier points
                    jitter=0.3,
                    pointpos=0,
                    line=dict(width=2),
                    fillcolor='rgba(255,0,0,0.1)',
                    whiskerwidth=0.8,
                    # Optional: provide pre-computed statistics for optimization with large datasets
                    q1=[q1],
                    median=[median],
                    q3=[q3],
                    lowerfence=[whisker_min],
                    upperfence=[whisker_max]
                ),
                row=1, col=1
            )
            
            # Calculate histogram bins
            data = df[col].dropna()
            if len(data) > 0:
                # Use Sturges' formula for bin count
                bin_count = int(np.ceil(np.log2(len(data))) + 1)
                bin_count = min(50, max(10, bin_count))  # Keep bins reasonable
                
                hist, bin_edges = np.histogram(data, bins=bin_count, density=True)
                
                # Use the center of each bin for x values
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                fig.add_trace(
                    go.Bar(
                        x=bin_centers,
                        y=hist,
                        name=col,
                        marker_color='blue',
                        opacity=0.7,
                        visible=True
                    ),
                    row=1, col=2
                )
                
                # Add normal distribution curve if appropriate
                mean = stats_info[col]['mean']
                std = stats_info[col]['std_dev']
                
                if pd.notna(mean) and pd.notna(std) and std > 0:
                    x_range = np.linspace(min(bin_edges), max(bin_edges), 100)
                    pdf = stats.norm.pdf(x_range, loc=mean, scale=std)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=x_range,
                            y=pdf,
                            mode='lines',
                            name='Normal Distribution',
                            line=dict(color='green', dash='dash'),
                            visible=True
                        ),
                        row=1, col=2
                    )
    
    # Add statistical annotations for the first column
    stats_annotations = [
        f"<b>Statistics:</b><br>" +
        f"Mean: {stats_info[first_col]['mean']:.2f}<br>" +
        f"Median: {stats_info[first_col]['median']:.2f}<br>" +
        f"StdDev: {stats_info[first_col]['std_dev']:.2f}<br>" +
        f"Skewness: {stats_info[first_col]['skewness']:.2f}"
    ]
    
    fig.add_annotation(
        xref="paper", yref="paper",
        x=1.02, y=0.4,
        xanchor="left", yanchor="middle",
        text=stats_annotations[0],
        showarrow=False,
        font=dict(size=12),
        bordercolor="black",
        borderwidth=1,
        borderpad=10,
        bgcolor="white",
        opacity=0.8
    )
    
    # Create dropdown menu buttons
    dropdown_buttons = []
    
    # Create dropdown entries for each numeric column
    for i, col in enumerate(numeric_cols):
        if df[col].count() == 0:
            continue
            
        # Improved box plot data preparation
        data = df[col].dropna()
        q1 = np.percentile(data, 25)
        median = np.percentile(data, 50)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        whisker_min = max(np.min(data), q1 - 1.5 * iqr)
        whisker_max = min(np.max(data), q3 + 1.5 * iqr)
        
        # Pre-calculate histogram data
        bin_count = int(np.ceil(np.log2(len(data))) + 1) if len(data) > 0 else 10
        bin_count = min(50, max(10, bin_count))
        
        hist_data = {}
        if len(data) > 0:
            hist, bin_edges = np.histogram(data, bins=bin_count, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            hist_data = {'centers': bin_centers, 'heights': hist, 'edges': bin_edges}
            
        # Normal distribution data if applicable
        mean = stats_info[col]['mean']
        std = stats_info[col]['std_dev']
        
        normal_curve = None
        if pd.notna(mean) and pd.notna(std) and std > 0 and len(data) > 0:
            x_range = np.linspace(min(bin_edges), max(bin_edges), 100)
            pdf = stats.norm.pdf(x_range, loc=mean, scale=std)
            normal_curve = {'x': x_range, 'y': pdf}
        
        # Create annotation text
        annotation_text = (
            f"<b>Statistics:</b><br>" +
            f"Mean: {stats_info[col]['mean']:.2f}<br>" +
            f"Median: {stats_info[col]['median']:.2f}<br>" +
            f"StdDev: {stats_info[col]['std_dev']:.2f}<br>" +
            f"Skewness: {stats_info[col]['skewness']:.2f}"
        )
        
        # Update plot title
        title_update = f"Data Analysis for {col}"
        
        # Create the dropdown button
        dropdown_buttons.append(
            dict(
                method='update',
                label=col,
                args=[
                    {
                        # Update boxplot data with improved configuration
                        'y': [data],
                        'q1': [[q1]],
                        'median': [[median]],
                        'q3': [[q3]],
                        'lowerfence': [[whisker_min]],
                        'upperfence': [[whisker_max]],
                        'name': [[col]],
                        
                        # Update histogram data (second trace)
                        'x': [None, hist_data.get('centers', []), normal_curve.get('x', []) if normal_curve else []],
                        'y': [None, hist_data.get('heights', []), normal_curve.get('y', []) if normal_curve else []]
                    },
                    {
                        # Replace the entire annotations array
                        "annotations": [
                            dict(
                                xref="paper", yref="paper",
                                x=1.02, y=0.4,
                                xanchor="left", yanchor="middle",
                                text=annotation_text,
                                showarrow=False,
                                font=dict(size=12),
                                bordercolor="black",
                                borderwidth=1,
                                borderpad=10,
                                bgcolor="white",
                                opacity=0.8
                            )
                        ],
                        # Update title
                        'title': title_update
                    }
                ]
            )
        )
    
    # Update layout
    fig.update_layout(
        title=f"Data Analysis for {first_col}",
        showlegend=True,
        title_x=0.5,
        margin=dict(r=200),
        height=600,
        updatemenus=[
            dict(
                buttons=dropdown_buttons,
                direction="down",
                showactive=True,
                x=0.5,
                y=1.1,
                xanchor="center",
                yanchor="top"
            )
        ]
    )
    
    # Update axes with correct labels
    fig.update_xaxes(title_text="Variable Value", row=1, col=2)
    fig.update_yaxes(title_text="Variable Value", row=1, col=1)
    fig.update_yaxes(title_text="Frequency Density", row=1, col=2)
    
    # Show the figure
    fig.show()
    
    return summary_df, df_clean