"""
MMIDNet model architecture implementation.

This module implements the MMIDNet architecture from the research paper for human
identification using point cloud data. The architecture consists of:

1. Transform Block (T-Net): Learns spatial transformations for rotation invariance
2. Residual CNN Block: Extracts point-wise features with residual connections
3. Global Max Pooling: Achieves permutation invariance
4. Bi-LSTM Block: Captures temporal relationships across frames
5. Dense Block: Final classification layers

The implementation is adapted for processing synthetic radar-like data generated
from FAUST 3D mesh scans, with 5 input channels: (x, y, z, velocity, snr).
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_tnet(input_shape, num_features=3, name='tnet'):
    """
    Create T-Net (Transform Network) for learning spatial transformations.
    
    T-Net learns a 3×3 transformation matrix to make the model rotation-invariant.
    This is critical for point cloud processing where data can be in any orientation.
    
    Architecture: Conv1D → BatchNorm → Dropout → GlobalMaxPool → Dense → Transform Matrix
    
    Args:
        input_shape: Shape of input points (num_points, num_features)
        num_features: Number of spatial features (3 for x, y, z)
        name: Layer name prefix
        
    Returns:
        Keras model that outputs (num_features, num_features) transformation matrix
    """
    inputs = layers.Input(shape=input_shape)
    
    # Feature extraction layers
    x = layers.Conv1D(64, 1, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(1e-4),
                     kernel_initializer='he_normal',
                     name=f'{name}_conv1')(inputs)
    x = layers.BatchNormalization(name=f'{name}_bn1')(x)
    x = layers.Dropout(0.1, name=f'{name}_dropout1')(x)
    
    x = layers.Conv1D(128, 1, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(1e-4),
                     kernel_initializer='he_normal',
                     name=f'{name}_conv2')(x)
    x = layers.BatchNormalization(name=f'{name}_bn2')(x)
    x = layers.Dropout(0.1, name=f'{name}_dropout2')(x)
    
    x = layers.Conv1D(256, 1, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(1e-4),
                     kernel_initializer='he_normal',
                     name=f'{name}_conv3')(x)
    x = layers.BatchNormalization(name=f'{name}_bn3')(x)
    x = layers.GlobalMaxPooling1D(name=f'{name}_pool')(x)
    
    # Dense layers for transformation matrix
    x = layers.Dense(128, activation='relu',
                    kernel_regularizer=keras.regularizers.l2(1e-4),
                    kernel_initializer='he_normal',
                    name=f'{name}_fc1')(x)
    x = layers.BatchNormalization(name=f'{name}_bn4')(x)
    x = layers.Dropout(0.15, name=f'{name}_dropout3')(x)
    
    x = layers.Dense(64, activation='relu',
                    kernel_regularizer=keras.regularizers.l2(1e-4),
                    kernel_initializer='he_normal',
                    name=f'{name}_fc2')(x)
    x = layers.BatchNormalization(name=f'{name}_bn5')(x)
    
    # Output transformation matrix (initialized to identity)
    initializer = keras.initializers.Constant(np.eye(num_features).flatten())
    transform = layers.Dense(
        num_features * num_features,
        kernel_initializer='zeros',
        bias_initializer=initializer,
        name=f'{name}_transform'
    )(x)
    transform = layers.Reshape((num_features, num_features), name=f'{name}_reshape')(transform)
    
    return keras.Model(inputs=inputs, outputs=transform, name=name)


def transform_block(inputs, name='transform_block'):
    """
    Transform Block: Applies T-Net transformation to input point clouds.
    
    Only transforms spatial coordinates (x, y, z), preserving velocity and SNR channels.
    Applied per frame using TimeDistributed wrapper.
    
    Args:
        inputs: (batch, time, points, 5) - input sequences
        name: Layer name prefix
        
    Returns:
        transformed: (batch, time, points, 5) - transformed sequences
    """
    # Input is (batch, time, points, 5) - (x, y, z, velocity, snr)
    # Extract only xyz coordinates for T-Net transformation
    xyz = inputs[:, :, :, :3]  # (batch, time, points, 3)
    
    # Get static shape for T-Net (use shape if available, otherwise infer from tensor)
    if hasattr(inputs, 'shape') and inputs.shape[2] is not None:
        num_points = int(inputs.shape[2])
    else:
        # Fallback: use a known value or get from tensor
        num_points = 200  # Default from our data
    
    # Create T-Net (only transforms xyz coordinates)
    tnet = create_tnet(input_shape=(num_points, 3), num_features=3, name=f'{name}_tnet')
    
    # Apply T-Net to each frame (TimeDistributed)
    time_distributed_tnet = layers.TimeDistributed(tnet, name=f'{name}_td_tnet')
    transform_matrix = time_distributed_tnet(xyz)  # (batch, time, 3, 3)
    
    # Apply transformation to xyz coordinates using Lambda layer
    def apply_transform(args):
        xyz, transform = args
        # Get dynamic shapes
        xyz_shape = tf.shape(xyz)
        batch_size = xyz_shape[0]
        time_steps = xyz_shape[1]
        num_points = xyz_shape[2]
        num_features = xyz_shape[3]
    
        # Reshape for matrix multiplication: (batch*time, points, features)
        xyz_reshaped = tf.reshape(xyz, [-1, num_points, num_features])
        transform_reshaped = tf.reshape(transform, [-1, num_features, num_features])
        
        # Apply transformation: (batch*time, points, features)
        transformed_xyz = tf.matmul(xyz_reshaped, transform_reshaped)
        
        # Reshape back: (batch, time, points, features)
        transformed_xyz = tf.reshape(transformed_xyz, [batch_size, time_steps, num_points, num_features])
        return transformed_xyz
    
    transformed_xyz = layers.Lambda(
        apply_transform,
        name=f'{name}_apply_transform'
    )([xyz, transform_matrix])
    
    # Concatenate velocity and SNR back (T-Net only transforms spatial coordinates)
    velocity_snr = inputs[:, :, :, 3:5]  # (batch, time, points, 2)
    output = layers.Concatenate(axis=-1, name=f'{name}_concat')([transformed_xyz, velocity_snr])
    
    return output  # (batch, time, points, 5)


def residual_cnn_block(inputs, name='residual_cnn'):
    """
    Residual CNN Block: Extracts hierarchical point-wise features.
    
    Uses TimeDistributed Conv1D layers with residual connections to capture
    local geometric patterns while preserving gradients through skip connections.
    
    Architecture: 3 residual blocks with increasing filter sizes (64 → 128 → 256)
    
    Args:
        inputs: (batch, time, points, channels) - transformed sequences
        name: Layer name prefix
        
    Returns:
        features: (batch, time, points, 256) - extracted features
    """
    x = inputs
    
    # Residual block 1: 5 channels → 64 filters
    x1 = layers.TimeDistributed(
        layers.Conv1D(64, 1, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(1e-4),
                     kernel_initializer='he_normal',
                     name=f'{name}_conv1a'),
        name=f'{name}_td1a'
    )(x)
    x1 = layers.TimeDistributed(
        layers.BatchNormalization(name=f'{name}_bn1a'),
        name=f'{name}_td_bn1a'
    )(x1)
    x1 = layers.TimeDistributed(
        layers.Dropout(0.3, name=f'{name}_dropout1a'),
        name=f'{name}_td_dropout1a'
    )(x1)
    x1 = layers.TimeDistributed(
        layers.Conv1D(64, 1, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(1e-4),
                     kernel_initializer='he_normal',
                     name=f'{name}_conv1b'),
        name=f'{name}_td1b'
    )(x1)
    x1 = layers.TimeDistributed(
        layers.BatchNormalization(name=f'{name}_bn1b'),
        name=f'{name}_td_bn1b'
    )(x1)
    skip1 = layers.TimeDistributed(
        layers.Conv1D(64, 1,
                     kernel_regularizer=keras.regularizers.l2(1e-4),
                     kernel_initializer='he_normal',
                     name=f'{name}_skip1'),
        name=f'{name}_td_skip1'
    )(x)
    x = layers.Add(name=f'{name}_add1')([x1, skip1])
    x = layers.ReLU(name=f'{name}_relu1')(x)
    
    # Residual block 2: 64 → 128 filters
    x2 = layers.TimeDistributed(
        layers.Conv1D(128, 1, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(1e-4),
                     kernel_initializer='he_normal',
                     name=f'{name}_conv2a'),
        name=f'{name}_td2a'
    )(x)
    x2 = layers.TimeDistributed(
        layers.BatchNormalization(name=f'{name}_bn2a'),
        name=f'{name}_td_bn2a'
    )(x2)
    x2 = layers.TimeDistributed(
        layers.Dropout(0.3, name=f'{name}_dropout2a'),
        name=f'{name}_td_dropout2a'
    )(x2)
    x2 = layers.TimeDistributed(
        layers.Conv1D(128, 1, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(1e-4),
                     kernel_initializer='he_normal',
                     name=f'{name}_conv2b'),
        name=f'{name}_td2b'
    )(x2)
    x2 = layers.TimeDistributed(
        layers.BatchNormalization(name=f'{name}_bn2b'),
        name=f'{name}_td_bn2b'
    )(x2)
    skip2 = layers.TimeDistributed(
        layers.Conv1D(128, 1,
                     kernel_regularizer=keras.regularizers.l2(1e-4),
                     kernel_initializer='he_normal',
                     name=f'{name}_skip2'),
        name=f'{name}_td_skip2'
    )(x)
    x = layers.Add(name=f'{name}_add2')([x2, skip2])
    x = layers.ReLU(name=f'{name}_relu2')(x)
    
    # Residual block 3: 128 → 256 filters
    x3 = layers.TimeDistributed(
        layers.Conv1D(256, 1, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(1e-4),
                     kernel_initializer='he_normal',
                     name=f'{name}_conv3a'),
        name=f'{name}_td3a'
    )(x)
    x3 = layers.TimeDistributed(
        layers.BatchNormalization(name=f'{name}_bn3a'),
        name=f'{name}_td_bn3a'
    )(x3)
    x3 = layers.TimeDistributed(
        layers.Dropout(0.2, name=f'{name}_dropout3a'),
        name=f'{name}_td_dropout3a'
    )(x3)
    x3 = layers.TimeDistributed(
        layers.Conv1D(256, 1, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(1e-4),
                     kernel_initializer='he_normal',
                     name=f'{name}_conv3b'),
        name=f'{name}_td3b'
    )(x3)
    x3 = layers.TimeDistributed(
        layers.BatchNormalization(name=f'{name}_bn3b'),
        name=f'{name}_td_bn3b'
    )(x3)
    skip3 = layers.TimeDistributed(
        layers.Conv1D(256, 1,
                     kernel_regularizer=keras.regularizers.l2(1e-4),
                     kernel_initializer='he_normal',
                     name=f'{name}_skip3'),
        name=f'{name}_td_skip3'
    )(x)
    x = layers.Add(name=f'{name}_add3')([x3, skip3])
    x = layers.ReLU(name=f'{name}_relu3')(x)
    
    return x  # (batch, time, points, 256)


def temporal_attention_block(inputs, name='temporal_attention'):
    """
    Temporal Attention: Learns to focus on important frames in the sequence.
    
    Uses soft attention mechanism to weight frames by their importance for classification.
    
    Args:
        inputs: (batch, time, features) - LSTM output sequences
        name: Layer name prefix
        
    Returns:
        attended: (batch, time, features) - attention-weighted sequences
    """
    # inputs: (batch, time, features) - already has time dimension
    attention = layers.Dense(1, activation='tanh',
                           kernel_regularizer=keras.regularizers.l2(1e-4),
                           kernel_initializer='he_normal',
                           name=f'{name}_dense')(inputs)
    # attention shape: (batch, time, 1)
    attention = layers.Softmax(axis=1, name=f'{name}_softmax')(attention)  # Softmax over time
    attended = layers.Multiply(name=f'{name}_multiply')([inputs, attention])
    return attended  # (batch, time, features)


def bilstm_block(inputs, name='bilstm_block'):
    """
    Bi-LSTM Block: Captures temporal relationships across frames.
    
    Uses bidirectional LSTM to learn both forward and backward temporal patterns,
    followed by temporal attention to focus on important frames.
    
    Architecture: BiLSTM(96) → Attention → BiLSTM(96)
    
    Args:
        inputs: (batch, time, features) - pooled features from CNN
        name: Layer name prefix
        
    Returns:
        temporal_features: (batch, 192) - final temporal representation
    """
    # First Bi-LSTM layer
    bilstm1 = layers.Bidirectional(
        layers.LSTM(96, return_sequences=True,
                    kernel_regularizer=keras.regularizers.l2(1e-4),
                    recurrent_regularizer=keras.regularizers.l2(1e-4),
                    dropout=0.2,
                    recurrent_dropout=0.2,
                    kernel_initializer='he_normal',
                    name=f'{name}_lstm1'),
        name=f'{name}_bilstm1'
    )(inputs)  # (batch, time, 192)
    
    # Apply temporal attention
    attended = temporal_attention_block(bilstm1, name=f'{name}_attention')
    
    # Second Bi-LSTM layer
    bilstm2 = layers.Bidirectional(
        layers.LSTM(96, return_sequences=False,
                    kernel_regularizer=keras.regularizers.l2(1e-4),
                    recurrent_regularizer=keras.regularizers.l2(1e-4),
                    dropout=0.2,
                    recurrent_dropout=0.2,
                    kernel_initializer='he_normal',
                    name=f'{name}_lstm2'),
        name=f'{name}_bilstm2'
    )(attended)  # (batch, 192)
    
    return bilstm2


def dense_block(inputs, num_classes=6, name='dense_block'):
    """
    Dense Block: Final classification layers.
    
    Maps temporal features to class probabilities using fully connected layers.
    
    Architecture: Dense(384) → Dense(192) → Dense(64) → Dense(num_classes)
    
    Args:
        inputs: (batch, features) - temporal features from Bi-LSTM
        num_classes: Number of output classes
        name: Layer name prefix
        
    Returns:
        outputs: (batch, num_classes) - class probabilities (softmax)
    """
    x = layers.Dense(384,
                    kernel_regularizer=keras.regularizers.l2(1e-4),
                    kernel_initializer='he_normal',
                    name=f'{name}_dense1')(inputs)
    x = layers.BatchNormalization(name=f'{name}_bn1')(x)
    x = layers.ReLU(name=f'{name}_relu1')(x)
    x = layers.Dropout(0.4, name=f'{name}_dropout1')(x)
    
    x = layers.Dense(192,
                    kernel_regularizer=keras.regularizers.l2(1e-4),
                    kernel_initializer='he_normal',
                    name=f'{name}_dense2')(x)
    x = layers.BatchNormalization(name=f'{name}_bn2')(x)
    x = layers.ReLU(name=f'{name}_relu2')(x)
    x = layers.Dropout(0.4, name=f'{name}_dropout2')(x)
    
    x = layers.Dense(64,
                    kernel_regularizer=keras.regularizers.l2(1e-4),
                    kernel_initializer='he_normal',
                    name=f'{name}_dense3')(x)
    x = layers.BatchNormalization(name=f'{name}_bn3')(x)
    x = layers.ReLU(name=f'{name}_relu3')(x)
    x = layers.Dropout(0.3, name=f'{name}_dropout3')(x)
    
    outputs = layers.Dense(num_classes, activation='softmax', name=f'{name}_output')(x)
    
    return outputs


def build_mmidnet(input_shape=(30, 200, 5), num_classes=6):
    """
    Build complete MMIDNet model for human identification.
    
    Full pipeline: Transform → Residual CNN → Global Max Pooling → Bi-LSTM → Dense
    
    Args:
        input_shape: (time, points, channels) - default (30, 200, 5)
                    5 channels: (x, y, z, velocity, snr)
        num_classes: Number of output classes (default: 6 combined classes)
        
    Returns:
        model: Compiled Keras model ready for training
    """
    inputs = layers.Input(shape=input_shape, name='input')
    
    # Transform Block
    x = transform_block(inputs, name='transform_block')
    
    # Residual CNN Block
    x = residual_cnn_block(x, name='residual_cnn')
    
    # Global Max Pooling (TimeDistributed)
    x = layers.TimeDistributed(
        layers.GlobalMaxPooling1D(name='global_max_pool'),
        name='td_global_max_pool'
    )(x)
    
    # Bi-LSTM Block with attention
    x = bilstm_block(x, name='bilstm_block')
    
    # Dense Block
    outputs = dense_block(x, num_classes=num_classes, name='dense_block')
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='MMIDNet')
    return model
