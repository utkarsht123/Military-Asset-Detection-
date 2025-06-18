import pandas as pd
import plotly.express as px

def create_confidence_chart(confidences):
    """
    Create a horizontal bar chart of class confidences using Plotly.
    Args:
        confidences (dict): Dictionary of class names to confidence scores.
    Returns:
        plotly.graph_objs._figure.Figure: Plotly Figure object.
    """
    # Convert dictionary to DataFrame
    df = pd.DataFrame(list(confidences.items()), columns=["Class", "Confidence"])
    # Create horizontal bar chart
    fig = px.bar(
        df,
        x="Confidence",
        y="Class",
        orientation="h",
        title="Model Confidence by Class",
        text="Confidence"
    )
    # Hide legend, adjust layout
    fig.update_layout(showlegend=False, xaxis_range=[0, 1])
    fig.update_traces(texttemplate='%{x:.2f}', textposition='outside')
    return fig 