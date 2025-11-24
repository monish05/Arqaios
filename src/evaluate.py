"""
Evaluation utilities for MMIDNet.

This module provides functions for:
- Standard model evaluation (accuracy, loss)
- Sliding window evaluation (robustness testing)
- Per-class accuracy computation
- Confusion matrix and classification report generation

Sliding window evaluation creates multiple predictions from augmented sequences
and aggregates them for improved robustness, similar to the paper's approach.
"""

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


def sliding_window_predict(model, sequence, num_variations=5):
    """
    Apply sliding window prediction for improved robustness.
    
    Creates multiple predictions from slightly augmented versions of the sequence
    and averages the probabilities. This mimics the paper's sliding window approach.
    
    Args:
        model: Trained Keras model
        sequence: (time, points, channels) input sequence
        num_variations: Number of augmented variations to create
        
    Returns:
        aggregated_pred: (num_classes,) averaged class probabilities
    """
    predictions = []
    
    # Original sequence
    pred = model.predict(sequence[np.newaxis, :], verbose=0)[0]
    predictions.append(pred)
    
    # Create variations with slight augmentations
    for _ in range(num_variations - 1):
        # Add small noise to simulate different windows
        noise = np.random.normal(0, 0.001, sequence.shape)
        augmented = sequence + noise
        pred = model.predict(augmented[np.newaxis, :], verbose=0)[0]
        predictions.append(pred)
    
    # Aggregate (average probabilities)
    aggregated = np.mean(predictions, axis=0)
    return aggregated


def evaluate_with_sliding_window(model, X_test, y_test_onehot, num_variations=5):
    """
    Evaluate model using sliding window aggregation.
    
    Applies sliding window prediction to all test sequences and computes accuracy.
    This provides more robust evaluation by averaging predictions from multiple variations.
    
    Args:
        model: Trained Keras model
        X_test: Test sequences (N, time, points, channels)
        y_test_onehot: One-hot encoded test labels (N, num_classes)
        num_variations: Number of variations per sequence
        
    Returns:
        accuracy: Overall accuracy
        all_predictions: (N, num_classes) prediction probabilities
        y_pred_classes: (N,) predicted class indices
        y_true_classes: (N,) true class indices
    """
    all_predictions = []
    
    print("Evaluating with sliding window...")
    for i, sequence in enumerate(X_test):
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(X_test)} sequences")
        
        pred = sliding_window_predict(model, sequence, num_variations)
        all_predictions.append(pred)
    
    all_predictions = np.array(all_predictions)
    y_pred_classes = np.argmax(all_predictions, axis=1)
    y_true_classes = np.argmax(y_test_onehot, axis=1)
    
    accuracy = np.mean(y_pred_classes == y_true_classes)
    return accuracy, all_predictions, y_pred_classes, y_true_classes


def evaluate_model(model, X_test, y_test_onehot):
    """
    Standard model evaluation without sliding window.
    
    Computes test loss, accuracy, and predictions for baseline comparison.
    
    Args:
        model: Trained Keras model
        X_test: Test sequences (N, time, points, channels)
        y_test_onehot: One-hot encoded test labels (N, num_classes)
        
    Returns:
        test_loss: Test loss value
        test_accuracy: Test accuracy
        y_pred: (N, num_classes) prediction probabilities
        y_pred_classes: (N,) predicted class indices
        y_true_classes: (N,) true class indices
    """
    test_loss, test_accuracy = model.evaluate(X_test, y_test_onehot, verbose=0)
    
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test_onehot, axis=1)
    
    return test_loss, test_accuracy, y_pred, y_pred_classes, y_true_classes


def get_per_class_accuracy(y_true, y_pred, num_classes=6):
    """
    Calculate per-class accuracy for detailed performance analysis.
    
    Useful for identifying which classes the model struggles with.
    
    Args:
        y_true: True class labels (N,)
        y_pred: Predicted class labels (N,)
        num_classes: Total number of classes
        
    Returns:
        per_class_acc: Dictionary mapping class_id to {'accuracy': float, 'count': int}
    """
    per_class_acc = {}
    for class_id in range(num_classes):
        class_mask = y_true == class_id
        if np.sum(class_mask) > 0:
            class_accuracy = np.mean(y_pred[class_mask] == y_true[class_mask])
            count = np.sum(class_mask)
            per_class_acc[class_id] = {'accuracy': class_accuracy, 'count': count}
    return per_class_acc

