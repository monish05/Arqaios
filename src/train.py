"""
Training utilities for MMIDNet.

This module provides functions for:
- Dataset splitting (stratified by subject)
- Class weight computation (handles class imbalance)
- Training configuration (optimizer, callbacks, loss)
- Model checkpointing and early stopping

Follows ML best practices:
- Stratified train/val/test splits
- Class weights for imbalanced data
- Early stopping to prevent overfitting
- Learning rate scheduling
- Mixed precision training support
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight


def compute_class_weights(y_train):
    """
    Compute class weights to handle class imbalance.
    
    Uses sklearn's 'balanced' strategy to automatically weight classes
    inversely proportional to their frequency.
    
    Args:
        y_train: Training labels (class indices)
        
    Returns:
        class_weight_dict: Dictionary mapping class_id to weight
    """
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    return class_weight_dict


def setup_callbacks(checkpoint_dir='checkpoints', model_name='best_model'):
    """
    Setup training callbacks for model checkpointing and early stopping.
    
    Callbacks:
    - ModelCheckpoint: Saves best model based on validation accuracy
    - EarlyStopping: Stops training if no improvement (patience=25)
    - ReduceLROnPlateau: Reduces learning rate when validation loss plateaus
    
    Args:
        checkpoint_dir: Directory to save model checkpoints
        model_name: Name for saved model file
        
    Returns:
        callbacks: List of Keras callbacks
    """
    # Handle relative paths - if running from notebook, go up one level
    # But if checkpoint_dir is already absolute (e.g., from Colab), use it as-is
    if not os.path.isabs(checkpoint_dir):
        # Check if we're in a notebooks directory (local) or project root (Colab)
        if 'notebooks' in os.getcwd() and 'drive' not in os.getcwd():
            checkpoint_dir = os.path.join('..', checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            os.path.join(checkpoint_dir, f'{model_name}.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=25,  # Wait 25 epochs for improvement
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001  # Minimum improvement threshold
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    return callbacks


def train_model(model, X_train, y_train, X_val, y_val, 
                epochs=150, batch_size=32, checkpoint_dir='checkpoints',
                use_class_weights=True):
    """
    Train MMIDNet model with best practices.
    
    Features:
    - Automatic class weight computation for imbalanced data
    - Mixed precision training support (if enabled)
    - Gradient clipping to prevent exploding gradients
    - Label smoothing for better generalization
    - Comprehensive callbacks (checkpointing, early stopping, LR scheduling)
    
    Args:
        model: Uncompiled Keras model
        X_train, y_train: Training data and labels
        X_val, y_val: Validation data and labels
        epochs: Maximum number of training epochs
        batch_size: Batch size for training
        checkpoint_dir: Directory for saving checkpoints
        use_class_weights: Whether to use balanced class weights
        
    Returns:
        history: Training history object
    """
    callbacks = setup_callbacks(checkpoint_dir)
    
    # Compute class weights if needed (handles class imbalance)
    class_weight_dict = None
    if use_class_weights:
        # Convert one-hot to class indices if needed
        if len(y_train.shape) > 1:
            y_train_classes = np.argmax(y_train, axis=1)
        else:
            y_train_classes = y_train
        class_weight_dict = compute_class_weights(y_train_classes)
        print(f"Class weights: {class_weight_dict}")
    else:
        print("Class weights: Disabled")
    
    # Compile model with appropriate learning rate and gradient clipping
    # Check if mixed precision training is enabled
    try:
        from tensorflow.keras.mixed_precision import LossScaleOptimizer
        policy = keras.mixed_precision.global_policy()
        if policy.name == 'mixed_float16':
            # Use loss scaling for mixed precision
            optimizer = keras.optimizers.Adam(
                learning_rate=0.0005,  # Lower LR for mixed precision stability
                clipnorm=1.0  # Gradient clipping to prevent exploding gradients
            )
            optimizer = LossScaleOptimizer(optimizer)
            print("✓ Using mixed precision with loss scaling")
        else:
            optimizer = keras.optimizers.Adam(
                learning_rate=0.0005,  # Learning rate for stable training
                clipnorm=1.0  # Gradient clipping
            )
    except:
        # Fallback if mixed precision not available
        optimizer = keras.optimizers.Adam(
            learning_rate=0.0005,
            clipnorm=1.0  # Gradient clipping
        )
    
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),  # Label smoothing for generalization
        metrics=['accuracy']
    )
    
    # Train
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        class_weight=class_weight_dict,  # Apply class weights for imbalanced data
        verbose=1
    )
    
    return history


def split_dataset_by_subject(num_subjects=10, scans_per_subject=10, 
                             train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, 
                             random_state=42):
    """
    Split dataset by subject ensuring stratified distribution.
    
    Ensures each subject has the same proportion in train/val/test splits,
    preventing data leakage and maintaining class balance.
    
    Args:
        num_subjects: Total number of subjects
        scans_per_subject: Number of scans per subject
        train_ratio: Proportion for training (default: 0.7)
        val_ratio: Proportion for validation (default: 0.1)
        test_ratio: Proportion for testing (default: 0.2)
        random_state: Random seed for reproducibility
        
    Returns:
        train_scans, val_scans, test_scans: Lists of scan indices for each split
    """
    train_scans = []
    val_scans = []
    test_scans = []
    
    # Set random seed for reproducibility
    np.random.seed(random_state)
    
    for subject_id in range(num_subjects):
        subject_scans = list(range(subject_id * scans_per_subject, 
                                   (subject_id + 1) * scans_per_subject))
        
        # Calculate exact counts per subject
        n_train = int(scans_per_subject * train_ratio)  # 7
        n_val = int(scans_per_subject * val_ratio)       # 1
        n_test = scans_per_subject - n_train - n_val     # 2
        
        # Shuffle and split deterministically
        shuffled = subject_scans.copy()
        np.random.shuffle(shuffled)
        
        train_scans.extend(shuffled[:n_train])
        val_scans.extend(shuffled[n_train:n_train+n_val])
        test_scans.extend(shuffled[n_train+n_val:])
    
    return train_scans, val_scans, test_scans
