from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def get_evaluation_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics for classification tasks.
    
    Args:
        y_true: Ground truth labels
        y_pred: Model predictions
        
    Returns:
        dict: Dictionary containing accuracy, precision, recall, and f1-score
    """
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    
    # Return metrics as dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    return metrics 