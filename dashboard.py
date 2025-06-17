import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns

def create_dummy_data():
    """
    Create sample data for demonstration when no file is uploaded.
    """
    dummy_data = {
        'model_name': [
            'LogisticRegression', 
            'RandomForest', 
            'DecisionTree', 
            'SVM', 
            'NeuralNetwork'
        ],
        'accuracy': [0.85, 0.92, 0.78, 0.88, 0.94],
        'precision': [0.83, 0.91, 0.76, 0.86, 0.93],
        'recall': [0.84, 0.90, 0.77, 0.87, 0.92],
        'f1_score': [0.83, 0.90, 0.76, 0.86, 0.92],
        'latency_ms': [2.5, 15.2, 8.7, 45.1, 120.3],
        'memory_mb': [15.2, 45.8, 22.1, 67.3, 156.7]
    }
    return pd.DataFrame(dummy_data)

def create_f1_score_chart(df):
    """
    Create a bar chart comparing F1 scores across models.
    """
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('model_name:N', title='Model', sort='-y'),
        y=alt.Y('f1_score:Q', title='F1 Score', scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('f1_score:Q', scale=alt.Scale(scheme='viridis')),
        tooltip=['model_name', 'f1_score']
    ).properties(
        title='F1 Score Comparison Across Models',
        width=600,
        height=400
    )
    
    return chart

def create_latency_accuracy_scatter(df):
    """
    Create a scatter plot showing latency vs accuracy trade-off.
    """
    chart = alt.Chart(df).mark_circle(size=100).encode(
        x=alt.X('latency_ms:Q', title='Latency (ms)'),
        y=alt.Y('accuracy:Q', title='Accuracy', scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('f1_score:Q', scale=alt.Scale(scheme='plasma')),
        size=alt.Size('memory_mb:Q', title='Memory (MB)'),
        tooltip=[
            alt.Tooltip('model_name:N', title='Model'),
            alt.Tooltip('accuracy:Q', title='Accuracy', format='.3f'),
            alt.Tooltip('latency_ms:Q', title='Latency (ms)', format='.1f'),
            alt.Tooltip('memory_mb:Q', title='Memory (MB)', format='.1f')
        ]
    ).properties(
        title='Latency vs Accuracy Trade-off',
        width=600,
        height=400
    )
    
    return chart

def create_performance_heatmap(df):
    """
    Create a heatmap showing all performance metrics.
    """
    # Select numeric columns for heatmap
    numeric_cols = ['accuracy', 'precision', 'recall', 'f1_score']
    heatmap_data = df.set_index('model_name')[numeric_cols]
    
    # Create matplotlib heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', fmt='.3f', ax=ax)
    plt.title('Performance Metrics Heatmap')
    plt.tight_layout()
    
    return fig

def main():
    # Page configuration
    st.set_page_config(
        page_title="Model Performance Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    # Title and description
    st.title("📊 Model Performance Dashboard")
    st.markdown("Upload your experiment results CSV file to visualize model performance metrics.")
    
    # File uploader
    st.header("📁 Data Upload")
    uploaded_file = st.file_uploader(
        "Choose a CSV file with model performance results",
        type=['csv'],
        help="CSV should contain columns: model_name, accuracy, precision, recall, f1_score, latency_ms, memory_mb"
    )
    
    # Load data
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(df)} model results!")
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.info("Using sample data instead.")
            df = create_dummy_data()
    else:
        st.info("📋 No file uploaded. Displaying sample data.")
        df = create_dummy_data()
    
    # Display raw data
    st.header("📋 Raw Data")
    st.dataframe(df, use_container_width=True)
    
    # Summary statistics
    st.header("📈 Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Best F1 Score", f"{df['f1_score'].max():.3f}")
    with col2:
        st.metric("Best Accuracy", f"{df['accuracy'].max():.3f}")
    with col3:
        st.metric("Fastest Model", f"{df['latency_ms'].min():.1f} ms")
    with col4:
        st.metric("Most Memory Efficient", f"{df['memory_mb'].min():.1f} MB")
    
    # Visualizations
    st.header("📊 Performance Visualizations")
    
    # F1 Score Comparison
    st.subheader("F1 Score Comparison")
    f1_chart = create_f1_score_chart(df)
    st.altair_chart(f1_chart, use_container_width=True)
    
    # Latency vs Accuracy Scatter
    st.subheader("Latency vs Accuracy Trade-off")
    scatter_chart = create_latency_accuracy_scatter(df)
    st.altair_chart(scatter_chart, use_container_width=True)
    
    # Performance Heatmap
    st.subheader("Performance Metrics Heatmap")
    heatmap_fig = create_performance_heatmap(df)
    st.pyplot(heatmap_fig)
    
    # Model Rankings
    st.header("🏆 Model Rankings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top Models by F1 Score")
        top_f1 = df.nlargest(3, 'f1_score')[['model_name', 'f1_score']]
        st.dataframe(top_f1, use_container_width=True)
    
    with col2:
        st.subheader("Fastest Models")
        fastest = df.nsmallest(3, 'latency_ms')[['model_name', 'latency_ms']]
        st.dataframe(fastest, use_container_width=True)
    
    # Detailed Analysis
    st.header("🔍 Detailed Analysis")
    
    # Model selection for detailed view
    selected_model = st.selectbox(
        "Select a model for detailed analysis:",
        df['model_name'].tolist()
    )
    
    if selected_model:
        model_data = df[df['model_name'] == selected_model].iloc[0]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Accuracy", f"{model_data['accuracy']:.3f}")
            st.metric("Precision", f"{model_data['precision']:.3f}")
        
        with col2:
            st.metric("Recall", f"{model_data['recall']:.3f}")
            st.metric("F1 Score", f"{model_data['f1_score']:.3f}")
        
        with col3:
            st.metric("Latency", f"{model_data['latency_ms']:.1f} ms")
            st.metric("Memory Usage", f"{model_data['memory_mb']:.1f} MB")
    
    # Footer
    st.markdown("---")
    st.markdown("*Dashboard created for Military Asset Detection Model Evaluation*")

if __name__ == "__main__":
    main() 