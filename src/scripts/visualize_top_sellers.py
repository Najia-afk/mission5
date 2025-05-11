import plotly.express as px
import pandas as pd

def visualize_top_sellers(df_sellers, n_sellers=30, id_column='seller_id', value_column='total_revenue'):
    """
    Creates a horizontal bar chart visualization of the top sellers by revenue,
    ordered from highest to lowest revenue.
    
    Parameters:
    -----------
    df_sellers : pandas.DataFrame
        DataFrame containing seller information with columns for ID and revenue
    n_sellers : int, optional (default=20)
        Number of top sellers to include in the visualization
    id_column : str, optional (default='seller_id')
        Name of the column containing seller IDs
    value_column : str, optional (default='total_revenue')
        Name of the column containing revenue values
        
    Returns:
    --------
    plotly.graph_objs._figure.Figure
        A plotly figure object with the bar chart
    """
    # Validate input
    if not isinstance(df_sellers, pd.DataFrame) or df_sellers.empty:
        raise ValueError("A non-empty pandas DataFrame must be provided")
    
    if id_column not in df_sellers.columns or value_column not in df_sellers.columns:
        raise ValueError(f"DataFrame must contain columns: {id_column} and {value_column}")
    
    # Get top n sellers and sort by revenue (descending)
    top_sellers = df_sellers.head(n_sellers).copy()
    top_sellers = top_sellers.sort_values(by=value_column, ascending=False)
    
    # Create shortened seller IDs for better readability
    top_sellers[f'{id_column}_short'] = top_sellers[id_column].str[:8] + '...'
    
    # Create a horizontal bar chart
    fig = px.bar(
        top_sellers,
        y=f'{id_column}_short',
        x=value_column,
        orientation='h',
        title=f'Top {n_sellers} Sellers by Revenue',
        hover_data=[id_column, value_column],
        labels={
            value_column: 'Revenue (BRL)', 
            f'{id_column}_short': 'Seller ID'
        },
        color=value_column,
        color_continuous_scale='Viridis'
    )
    
    # Add revenue values on the bars
    fig.update_traces(
        texttemplate='%{x:,.0f}',
        textposition='outside',
    )
    
    # Improve the layout
    fig.update_layout(
        xaxis_title='Revenue (BRL)',
        yaxis_title='',
        yaxis={'categoryorder': 'total ascending'},  # Put highest revenue at the top
        uniformtext_minsize=8,
        uniformtext_mode='hide',
        height=max(500, n_sellers * 25)  # Adjust height based on number of sellers
    )
    
    return fig