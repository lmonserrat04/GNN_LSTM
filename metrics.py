from sklearn.metrics import confusion_matrix, roc_auc_score


def calculate_metrics(y_true, y_pred, y_pred_prob):
    """
    Calculate key evaluation metrics for binary classification tasks.
    """
    print("   ↳ Calculando métricas...")
    # Compute confusion matrix and unpack values
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Calculate metrics with safeguards against division by zero
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    auc = roc_auc_score(y_true, y_pred_prob)

    return accuracy, sensitivity, precision, specificity, auc, cm

def print_metrics(split, dataset_type, accuracy, sensitivity, precision, specificity, auc, cm):
    """
    Display evaluation metrics for a specific data split and dataset type.
    """
    print(f"   ↳ Mostrando métricas para {dataset_type}...")
    print(f"\n{dataset_type.capitalize()} Metrics for Split {split + 1}:")
    print(f"  Accuracy: {accuracy * 100:.2f}%")
    print(f"  Sensitivity (Recall): {sensitivity * 100:.2f}%")
    print(f"  Precision: {precision * 100:.2f}%")
    print(f"  Specificity: {specificity * 100:.2f}%")
    print(f"  AUC-ROC Score: {auc * 100:.2f}%")
    print(f"  Confusion Matrix:\n{cm}")
