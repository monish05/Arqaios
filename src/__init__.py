"""
MMIDNet: Human Identification from Point Clouds

A proof-of-concept implementation of the MMIDNet architecture for human identification
using point cloud data. This package provides:

- Data preprocessing: Loading PLY files, FPS sampling, temporal sequence generation
- Model architecture: MMIDNet with T-Net, Residual CNN, Bi-LSTM components
- Training utilities: Stratified splitting, class weights, callbacks
- Evaluation: Standard and sliding window evaluation metrics

Designed to work with MPI-FAUST dataset to simulate mmWave radar point cloud data.
"""

__version__ = "1.0.0"

