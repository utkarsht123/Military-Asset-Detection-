import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from cpu_model_optimization import SimpleCNN
from deployment_utils import load_model, get_model_size_mb
from focused_eval import get_evaluation_metrics
from cpu_performance_tests import profile_inference_speed

def create_test_data(n_samples=500, n_features=32*32*3):
    """
    Create test data for benchmarking.
    
    Args:
        n_samples: Number of test samples
        n_features: Number of features (flattened image size)
        
    Returns:
        tuple: (X_test, y_test) - test data and labels
    """
    # Create synthetic classification data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.8),
        n_redundant=int(n_features * 0.2),
        n_classes=3,
        random_state=42
    )
    
    # Reshape to image format (batch_size, channels, height, width)
    X = X.reshape(-1, 3, 32, 32)
    
    # Split into train/test (we only need test for benchmarking)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    return X_test, y_test

def benchmark_model(model, X_test, y_test, model_name):
    """
    Run comprehensive benchmarks on a single model.
    
    Args:
        model: PyTorch model to benchmark
        X_test: Test data
        y_test: Test labels
        model_name: Name of the model for reporting
        
    Returns:
        dict: Dictionary with benchmark results
    """
    print(f"Benchmarking {model_name}...")
    
    # Convert numpy arrays to torch tensors
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Set model to evaluation mode
    model.eval()
    
    # 1. Accuracy Benchmark
    print(f"  - Testing accuracy...")
    with torch.no_grad():
        predictions = model(X_test_tensor)
        predicted_labels = torch.argmax(predictions, dim=1).numpy()
    
    # Get evaluation metrics
    metrics = get_evaluation_metrics(y_test, predicted_labels)
    
    # 2. Latency Benchmark
    print(f"  - Testing latency...")
    latency_ms = profile_inference_speed(model, X_test_tensor, n_runs=50)
    
    # 3. Model Size Benchmark
    print(f"  - Calculating model size...")
    model_size_mb = get_model_size_mb(f"models/{model_name.lower()}_model.pth")
    
    return {
        'model_name': model_name,
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1_score': metrics['f1_score'],
        'latency_ms': latency_ms,
        'model_size_mb': model_size_mb,
        'predictions': predicted_labels
    }

def generate_confusion_matrices(results, y_test, save_path="confusion_matrices.png"):
    """
    Generate and save confusion matrices for both models.
    
    Args:
        results: Dictionary containing benchmark results
        y_test: True labels
        save_path: Path to save the confusion matrix plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot confusion matrix for FP32 model
    disp_fp32 = ConfusionMatrixDisplay.from_predictions(
        y_test,
        results['FP32']['predictions'],
        ax=axes[0],
        colorbar=False
    )
    axes[0].set_title(f"FP32 Model Confusion Matrix\nAccuracy: {results['FP32']['accuracy']:.3f}")
    
    # Plot confusion matrix for INT8 model
    disp_int8 = ConfusionMatrixDisplay.from_predictions(
        y_test,
        results['INT8']['predictions'],
        ax=axes[1],
        colorbar=False
    )
    axes[1].set_title(f"INT8 Model Confusion Matrix\nAccuracy: {results['INT8']['accuracy']:.3f}")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrices saved to {save_path}")

def print_benchmark_report(results):
    """
    Print a formatted benchmark report.
    
    Args:
        results: Dictionary containing benchmark results for both models
    """
    print("\n" + "=" * 80)
    print("DEPLOYMENT BENCHMARK RESULTS")
    print("=" * 80)
    
    # Create comparison table
    comparison_data = []
    for model_type in ['FP32', 'INT8']:
        result = results[model_type]
        comparison_data.append({
            'Model': result['model_name'],
            'Accuracy': f"{result['accuracy']:.4f}",
            'F1-Score': f"{result['f1_score']:.4f}",
            'Latency (ms)': f"{result['latency_ms']:.2f}",
            'Model Size (MB)': f"{result['model_size_mb']:.2f}",
            'Throughput (pred/s)': f"{1000/result['latency_ms']:.1f}"
        })
    
    # Display table
    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))
    
    # Calculate improvements
    fp32 = results['FP32']
    int8 = results['INT8']
    
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    
    # Accuracy comparison
    acc_diff = int8['accuracy'] - fp32['accuracy']
    print(f"Accuracy Difference (INT8 - FP32): {acc_diff:+.4f} ({acc_diff*100:+.2f}%)")
    
    # Speed comparison
    speed_improvement = fp32['latency_ms'] / int8['latency_ms']
    print(f"Speed Improvement: {speed_improvement:.2f}x faster")
    
    # Size comparison
    size_reduction = fp32['model_size_mb'] / int8['model_size_mb']
    print(f"Size Reduction: {size_reduction:.2f}x smaller ({size_reduction*100:.1f}% reduction)")
    
    # Throughput comparison
    throughput_improvement = int8['latency_ms'] / fp32['latency_ms']
    print(f"Throughput Improvement: {throughput_improvement:.2f}x more predictions/second")
    
    # Overall assessment
    print("\n" + "=" * 80)
    print("OVERALL ASSESSMENT")
    print("=" * 80)
    
    if abs(acc_diff) < 0.01:
        print("✓ Quantization maintained accuracy (minimal degradation)")
    elif acc_diff > 0:
        print("✓ Quantization improved accuracy (unusual but possible)")
    else:
        print("⚠ Quantization reduced accuracy (trade-off for efficiency)")
    
    if speed_improvement > 1.5:
        print("✓ Significant speed improvement achieved")
    else:
        print("⚠ Limited speed improvement")
    
    if size_reduction > 2.0:
        print("✓ Significant size reduction achieved")
    else:
        print("⚠ Limited size reduction")

def main():
    """
    Main function to run the comprehensive performance analysis.
    """
    print("=" * 80)
    print("COMPREHENSIVE MODEL PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    # Step 1: Load models
    print("\n1. Loading models...")
    try:
        fp32_model = load_model(SimpleCNN, "models/fp32_model.pth", num_classes=3)
        print("✓ FP32 model loaded successfully")
        
        int8_model = load_model(SimpleCNN, "models/int8_model.pth",is_quantized=True, num_classes=3)
        print("✓ INT8 model loaded successfully")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please run model_variants.py first to create the model files.")
        return
    
    # Step 2: Prepare test data
    print("\n2. Preparing test data...")
    X_test, y_test = create_test_data(n_samples=500)
    print(f"✓ Test data created: {X_test.shape[0]} samples")
    
    # Step 3: Run benchmarks
    print("\n3. Running benchmarks...")
    results = {}
    
    # Benchmark FP32 model
    results['FP32'] = benchmark_model(fp32_model, X_test, y_test, "FP32")
    
    # Benchmark INT8 model
    results['INT8'] = benchmark_model(int8_model, X_test, y_test, "INT8")
    
    # Step 4: Generate confusion matrices
    print("\n4. Generating confusion matrices...")
    generate_confusion_matrices(results, y_test)
    
    # Step 5: Print comprehensive report
    print("\n5. Generating benchmark report...")
    print_benchmark_report(results)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    main()
