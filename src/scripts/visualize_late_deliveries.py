import pandas as pd
import plotly.express as px

def visualize_late_deliveries(df_deliveries):
    """
    Create a stacked bar chart visualization of late deliveries by month and delay category.
    
    Parameters:
    -----------
    df_deliveries : pandas.DataFrame
        DataFrame containing late delivery information with columns:
        order_purchase_timestamp, order_estimated_delivery_date, order_delivered_customer_date
        
    Returns:
    --------
    plotly.graph_objs._figure.Figure
        A plotly figure object with the stacked bar chart
    """
    # Make a copy to avoid modifying the original dataframe
    df = df_deliveries.copy()
    
    # Convert string dates to datetime objects
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    df['order_estimated_delivery_date'] = pd.to_datetime(df['order_estimated_delivery_date'])
    df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])

    # Calculate delay in days (delivered - estimated)
    df['delay_days'] = (df['order_delivered_customer_date'] - 
                         df['order_estimated_delivery_date']).dt.days

    # Create delay categories
    def categorize_delay(days):
        if days < 3:
            return "No Delay"
        elif days <= 4:
            return "3-4 days"
        elif days <= 5:
            return "4-5 days"
        elif days <= 7:
            return "5-7 days"
        elif days <= 10:
            return "7-10 days"
        else:
            return "10+ days"

    df['delay_category'] = df['delay_days'].apply(categorize_delay)

    # Extract month from purchase timestamp for grouping
    df['purchase_month'] = df['order_purchase_timestamp'].dt.to_period('M')

    # Count deliveries by month and delay category
    monthly_delays = df.groupby(['purchase_month', 'delay_category']).size().reset_index(name='count')
    monthly_delays['purchase_month'] = monthly_delays['purchase_month'].astype(str)

    # Create stacked bar chart
    fig = px.bar(monthly_delays, x='purchase_month', y='count', color='delay_category',
             title='Evolution of Late Deliveries Over Time',
             labels={'purchase_month': 'Month', 'count': 'Number of Late Deliveries', 'delay_category': 'Delay Category'},
             category_orders={"delay_category": ["No Delay","3-4 days", "4-5 days", "5-7 days", "7-10 days", "10+ days"]},
             color_discrete_sequence=px.colors.qualitative.Set2)

    # Hide No Delay from legend
    for trace in fig.data:
        if trace.name == "No Delay":
            trace.visible = "legendonly"

    fig.update_layout(xaxis_title='Order Purchase Month',
                      yaxis_title='Number of Late Deliveries',
                      legend_title='Delay Category',
                      barmode='stack')
    
    return fig