# simple_evaluation.py
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def get_basic_metrics(y_true, y_pred):
    """
    Calculates basic classification metrics.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }