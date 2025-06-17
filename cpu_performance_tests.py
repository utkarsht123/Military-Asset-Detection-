import time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from memory_profiler import profile

def profile_inference_speed(model, X_test, n_runs=100):
    """
    Measure the average inference time per prediction.
    
    Args:
        model: Trained scikit-learn model
        X_test: Test data for inference
        n_runs: Number of times to run inference for averaging
        
    Returns:
        float: Average latency in milliseconds
    """
    # Warm up the model (first few predictions can be slower)
    _ = model.predict(X_test[:10])
    
    # Measure inference time
    start_time = time.time()
    
    for _ in range(n_runs):
        _ = model.predict(X_test)
    
    end_time = time.time()
    
    # Calculate average time per run
    total_time = end_time - start_time
    avg_time_per_run = total_time / n_runs
    
    # Convert to milliseconds
    avg_latency_ms = avg_time_per_run * 1000
    
    return avg_latency_ms

@profile
def _memory_profiled_inference(model, X_test):
    """
    Inner function decorated with @profile for memory profiling.
    This function will be called by profile_memory_usage.
    """
    return model.predict(X_test)

def profile_memory_usage(model, X_test):
    """
    Measure peak memory usage during inference.
    
    Note: This function uses the @profile decorator which prints to stdout.
    For detailed memory profiling, run this script with:
    python -m memory_profiler cpu_performance_tests.py
    
    Args:
        model: Trained scikit-learn model
        X_test: Test data for inference
        
    Returns:
        None: Memory usage is printed to stdout
    """
    print("Running memory profiling...")
    print("Note: For detailed memory analysis, run: python -m memory_profiler cpu_performance_tests.py")
    print("-" * 50)
    
    # Call the profiled function
    _ = _memory_profiled_inference(model, X_test)
    
    print("-" * 50)
    print("Memory profiling completed.")

def create_test_data(n_samples=1000, n_features=20):
    """
    Create synthetic test data for performance testing.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        
    Returns:
        tuple: (X_test, y_test)
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42
    )
    
    # Split to get test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_test, y_test

def main():
    """
    Example usage of the performance profiling functions.
    """
    print("=" * 60)
    print("CPU PERFORMANCE PROFILING UTILITY")
    print("=" * 60)
    
    # Create test data
    print("Creating test data...")
    X_test, y_test = create_test_data(n_samples=1000, n_features=20)
    print(f"Test data shape: {X_test.shape}")
    print()
    
    # Create and train a dummy model
    print("Training LogisticRegression model...")
    X_train, _, y_train, _ = train_test_split(
        make_classification(n_samples=1000, n_features=20, random_state=42)[0],
        make_classification(n_samples=1000, n_features=20, random_state=42)[1],
        test_size=0.2, random_state=42
    )
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    print("✓ Model trained successfully")
    print()
    
    # Profile inference speed
    print("=" * 40)
    print("INFERENCE SPEED PROFILING")
    print("=" * 40)
    
    latency = profile_inference_speed(model, X_test, n_runs=100)
    print(f"Average inference latency: {latency:.2f} ms")
    print(f"Throughput: {1000/latency:.1f} predictions/second")
    print()
    
    # Profile memory usage
    print("=" * 40)
    print("MEMORY USAGE PROFILING")
    print("=" * 40)
    
    profile_memory_usage(model, X_test)
    
    # Additional model comparison
    print("\n" + "=" * 40)
    print("MODEL COMPARISON")
    print("=" * 40)
    
    models = {
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'RandomForestClassifier': RandomForestClassifier(n_estimators=50, random_state=42)
    }
    
    results = []
    for model_name, model_instance in models.items():
        print(f"Testing {model_name}...")
        
        # Train the model
        model_instance.fit(X_train, y_train)
        
        # Profile speed
        latency = profile_inference_speed(model_instance, X_test, n_runs=50)
        
        results.append({
            'Model': model_name,
            'Latency (ms)': f"{latency:.2f}",
            'Throughput (pred/s)': f"{1000/latency:.1f}"
        })
        
        print(f"✓ {model_name} completed")
    
    # Display comparison
    print("\nPerformance Comparison:")
    comparison_df = pd.DataFrame(results)
    print(comparison_df.to_string(index=False))

# Example of how to use memory profiling (commented out)
"""
# To run detailed memory profiling, uncomment this section and run:
# python -m memory_profiler cpu_performance_tests.py

@profile
def detailed_memory_profiling_example():
    # Create data and model
    X_test, _ = create_test_data()
    model = LogisticRegression(random_state=42)
    
    # This will show detailed memory usage
    predictions = model.predict(X_test)
    return predictions

if __name__ == "__main__":
    detailed_memory_profiling_example()
"""

if __name__ == "__main__":
    # Import here to avoid circular imports
    from sklearn.model_selection import train_test_split
    import pandas as pd
    
    main() 