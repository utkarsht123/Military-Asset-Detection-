import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from focused_eval import get_evaluation_metrics

def main():
    # Create synthetic classification dataset (same as baseline script)
    print("Creating synthetic dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42
    )
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Dataset shape: {X.shape}")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print()
    
    # Define the RandomForest model
    rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
    
    # ABLATION STUDY: Full Model vs Ablated Model
    
    print("=" * 60)
    print("ABLATION STUDY: FEATURE SELECTION IMPACT")
    print("=" * 60)
    
    # 1. Full Model (Control) - All features
    print("\n1. TRAINING FULL MODEL (All 20 features)...")
    full_model = rf_model.fit(X_train, y_train)
    full_pred = full_model.predict(X_test)
    full_metrics = get_evaluation_metrics(y_test, full_pred)
    
    print("✓ Full model training completed")
    
    # 2. Ablated Model (Test) - Only first 10 features
    print("\n2. TRAINING ABLATED MODEL (First 10 features only)...")
    X_train_ablated = X_train[:, :10]  # Select only first 10 features
    X_test_ablated = X_test[:, :10]    # Select only first 10 features
    
    ablated_model = RandomForestClassifier(n_estimators=50, random_state=42)
    ablated_model.fit(X_train_ablated, y_train)
    ablated_pred = ablated_model.predict(X_test_ablated)
    ablated_metrics = get_evaluation_metrics(y_test, ablated_pred)
    
    print("✓ Ablated model training completed")
    
    # 3. Results Comparison
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    
    # Create comparison table
    comparison_data = {
        'Model': ['Full Model (20 features)', 'Ablated Model (10 features)'],
        'Accuracy': [f"{full_metrics['accuracy']:.4f}", f"{ablated_metrics['accuracy']:.4f}"],
        'Precision': [f"{full_metrics['precision']:.4f}", f"{ablated_metrics['precision']:.4f}"],
        'Recall': [f"{full_metrics['recall']:.4f}", f"{ablated_metrics['recall']:.4f}"],
        'F1-Score': [f"{full_metrics['f1_score']:.4f}", f"{ablated_metrics['f1_score']:.4f}"]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # 4. Performance Analysis
    print("\n" + "=" * 60)
    print("PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    # Calculate performance differences
    full_acc = full_metrics['accuracy']
    ablated_acc = ablated_metrics['accuracy']
    acc_diff = full_acc - ablated_acc
    
    full_f1 = full_metrics['f1_score']
    ablated_f1 = ablated_metrics['f1_score']
    f1_diff = full_f1 - ablated_f1
    
    print(f"Feature Reduction: 20 features → 10 features (50% reduction)")
    print(f"Accuracy Impact: {acc_diff:+.4f} ({acc_diff*100:+.2f}%)")
    print(f"F1-Score Impact: {f1_diff:+.4f} ({f1_diff*100:+.2f}%)")
    
    # Determine if ablation had significant impact
    if abs(acc_diff) < 0.01:
        print("\n📊 CONCLUSION: Minimal impact from feature reduction")
    elif acc_diff > 0:
        print("\n📊 CONCLUSION: Full model performs better - removed features were important")
    else:
        print("\n📊 CONCLUSION: Ablated model performs better - removed features were noise")
    
    print(f"\nFeature efficiency: {ablated_acc:.4f} accuracy with {10} features vs {full_acc:.4f} with {20} features")

if __name__ == "__main__":
    main() 