import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from focused_eval import get_evaluation_metrics

def main():
    # Create synthetic classification dataset
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
    
    # Define baseline models
    models = {
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'DecisionTreeClassifier': DecisionTreeClassifier(max_depth=5, random_state=42),
        'RandomForestClassifier': RandomForestClassifier(n_estimators=50, random_state=42)
    }
    
    # Store results
    results = []
    
    # Train and evaluate each model
    print("Training and evaluating models...")
    print("-" * 50)
    
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get evaluation metrics
        metrics = get_evaluation_metrics(y_test, y_pred)
        
        # Store results
        results.append({
            'Model': model_name,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1-Score': f"{metrics['f1_score']:.4f}"
        })
        
        print(f"✓ {model_name} completed")
    
    # Create and display results table
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 60)
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # Find best model for each metric
    print("\n" + "=" * 60)
    print("BEST PERFORMING MODELS")
    print("=" * 60)
    
    # Convert string metrics back to float for comparison
    numeric_results = []
    for result in results:
        numeric_results.append({
            'Model': result['Model'],
            'Accuracy': float(result['Accuracy']),
            'Precision': float(result['Precision']),
            'Recall': float(result['Recall']),
            'F1-Score': float(result['F1-Score'])
        })
    
    numeric_df = pd.DataFrame(numeric_results)
    
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
        best_model = numeric_df.loc[numeric_df[metric].idxmax()]
        print(f"Best {metric}: {best_model['Model']} ({best_model[metric]:.4f})")

if __name__ == "__main__":
    main() 