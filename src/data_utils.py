"""
Data loading and preprocessing utilities for MPI-FAUST dataset.

This module implements data preprocessing to simulate mmWave radar point cloud data
from static 3D mesh scans. The preprocessing includes:
- Point cloud downsampling using Farthest Point Sampling (FPS)
- Synthetic temporal sequence generation (mimicking radar frames)
- Subject-specific characteristics (body scale, walking speed, SNR)
- Data augmentation (rotation, translation, noise)

The goal is to create a POC that demonstrates human identification using point clouds,
similar to the MMIDNet architecture from the research paper, but using FAUST data
instead of real radar data.
"""

import numpy as np
import os

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

try:
    from plyfile import PlyData
    HAS_PLYFILE = True
except ImportError:
    HAS_PLYFILE = False


def load_ply_file(ply_path):
    """
    Load PLY file and extract vertex coordinates.
    
    Supports both trimesh and plyfile libraries for maximum compatibility.
    
    Args:
        ply_path: Path to PLY file
        
    Returns:
        vertices: (N, 3) numpy array of x, y, z coordinates
        
    Raises:
        ValueError: If file cannot be loaded with either library
    """
    if HAS_TRIMESH:
        try:
            mesh = trimesh.load(ply_path)
            vertices = np.array(mesh.vertices)
            return vertices
        except Exception as e:
            if HAS_PLYFILE:
                pass
            else:
                raise ValueError(f"trimesh failed and plyfile not available: {e}")
    
    if HAS_PLYFILE:
        try:
            plydata = PlyData.read(ply_path)
            vertices = np.array([list(v) for v in plydata['vertex']])
            return vertices
        except Exception as e:
            raise ValueError(f"plyfile failed: {e}")
    
    raise ValueError(f"Could not load PLY file: {ply_path}")


def create_rotation_matrix_z(angle):
    """
    Create 3D rotation matrix around z-axis.
    
    Used for data augmentation to simulate different orientations.
    
    Args:
        angle: Rotation angle in radians
        
    Returns:
        rotation_matrix: (3, 3) rotation matrix
    """
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    return np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])


def farthest_point_sampling(points, num_samples, subject_id=None):
    """
    Farthest Point Sampling (FPS) - preserves body shape better than random sampling
    
    FPS ensures uniform coverage of the point cloud while preserving the overall
    body shape. This is critical for distinguishing between subjects.
    
    Made deterministic per subject: sorts points deterministically before FPS
    to ensure same subject always gets similar sampling.
    
    Args:
        points: (N, 3) point cloud
        num_samples: Number of points to sample
        subject_id: Optional subject ID for deterministic sorting (ensures same subject gets similar points)
    
    Returns:
        sampled_points: (num_samples, 3) sampled points
        indices: (num_samples,) indices of sampled points
    """
    if len(points) <= num_samples:
        return points.copy(), np.arange(len(points))
    
    # Make FPS deterministic per subject by sorting points consistently
    # This ensures same subject always gets similar point selection
    if subject_id is not None:
        # Create deterministic sort order based on coordinates + subject_id
        # Sort by x, then y, then z, with subject_id as tie-breaker seed
        sort_key = (points[:, 0] * 1000 + 
                   points[:, 1] * 100 + 
                   points[:, 2] * 10 + 
                   np.arange(len(points)) * 0.001)  # Use index as final tie-breaker
        sorted_indices = np.argsort(sort_key)
        points_sorted = points[sorted_indices]
        # Store original indices for mapping back
        original_to_sorted = {i: idx for idx, i in enumerate(sorted_indices)}
    else:
        points_sorted = points
        sorted_indices = np.arange(len(points))
        original_to_sorted = {i: i for i in range(len(points))}
    
    # Start with a point (use centroid or first point)
    indices_sorted = np.zeros(num_samples, dtype=np.int64)
    distances = np.ones(len(points_sorted)) * np.inf
    
    # First point: use centroid (center of mass) for better coverage
    centroid = np.mean(points_sorted, axis=0)
    dists_to_centroid = np.sum((points_sorted - centroid) ** 2, axis=1)
    first_idx = np.argmax(dists_to_centroid)  # Farthest from centroid
    indices_sorted[0] = first_idx
    
    # Iteratively select points farthest from already selected points
    for i in range(1, num_samples):
        # Update distances to the last selected point
        last_point = points_sorted[indices_sorted[i-1]]
        dists = np.sum((points_sorted - last_point) ** 2, axis=1)
        
        # Keep minimum distance to any selected point
        distances = np.minimum(distances, dists)
        
        # Select point with maximum minimum distance
        # If there are ties, use deterministic tie-breaking (first occurrence in sorted order)
        max_dist = np.max(distances)
        candidates = np.where(distances == max_dist)[0]
        if len(candidates) > 1:
            # Use first candidate (deterministic)
            indices_sorted[i] = candidates[0]
        else:
            indices_sorted[i] = np.argmax(distances)
    
    # Map back to original point indices
    if subject_id is not None:
        indices_original = np.array([sorted_indices[idx] for idx in indices_sorted])
    else:
        indices_original = indices_sorted
    
    return points[indices_original].copy(), indices_original


def get_subject_characteristics(subject_id):
    """
    Get subject-specific characteristics based on physical descriptions
    Returns: dict with walking_speed_range, body_scale, movement_consistency, base_snr
    
    Subject descriptions:
    0: young adult, slightly thin
    1: woman 30s, little fat
    2-3: old man 60s, active
    4: old woman 60s, weak
    5-6: young woman 20s, fit
    7: young man, muscular
    8: old woman, fat
    9: young man, little muscular
    """
    characteristics = {
        0: {  # Young adult, slightly thin
            'walking_speed_range': (0.025, 0.040),  # Fast, energetic
            'body_scale': 0.85,  # Smaller build (thin)
            'movement_consistency': 0.85,  # Consistent movement
            'base_snr': 20.0,  # Lower SNR (thin body = less reflective)
        },
        1: {  # Woman 30s, little fat
            'walking_speed_range': (0.008, 0.015),  # Slower pace
            'body_scale': 1.20,  # Larger build
            'movement_consistency': 0.65,  # Moderate consistency
            'base_snr': 32.0,  # Higher SNR (more reflective surface)
        },
        2: {  # Old man 60s, active
            'walking_speed_range': (0.018, 0.028),  # Moderate-fast
            'body_scale': 1.05,  # Medium-large build
            'movement_consistency': 0.70,  # Moderate consistency
            'base_snr': 26.0,  # Medium-high SNR
        },
        3: {  # Old man 60s, active (similar to subject 2)
            'walking_speed_range': (0.020, 0.030),  # Slightly faster
            'body_scale': 1.08,  # Slightly larger
            'movement_consistency': 0.72,  # Slightly more consistent
            'base_snr': 27.0,  # Slightly higher SNR
        },
        4: {  # Old woman 60s, weak
            'walking_speed_range': (0.005, 0.012),  # Very slow, weak
            'body_scale': 0.88,  # Small build (frail)
            'movement_consistency': 0.40,  # Inconsistent (weak)
            'base_snr': 18.0,  # Low SNR (weak signal)
        },
        5: {  # Young woman 20s, fit
            'walking_speed_range': (0.030, 0.045),  # Very fast, fit
            'body_scale': 0.95,  # Lean build
            'movement_consistency': 0.95,  # Very consistent
            'base_snr': 24.0,  # Medium SNR
        },
        6: {  # Young woman 20s, fit (similar to subject 5)
            'walking_speed_range': (0.028, 0.042),  # Fast, fit
            'body_scale': 0.98,  # Slightly larger than subject 5
            'movement_consistency': 0.92,  # Very consistent
            'base_snr': 25.0,  # Slightly higher SNR
        },
        7: {  # Young man, muscular
            'walking_speed_range': (0.035, 0.050),  # Very fast, strong
            'body_scale': 1.30,  # Large build (muscular)
            'movement_consistency': 0.90,  # Very consistent
            'base_snr': 35.0,  # Very high SNR (dense muscle, highly reflective)
        },
        8: {  # Old woman, fat
            'walking_speed_range': (0.006, 0.012),  # Very slow
            'body_scale': 1.25,  # Very large build
            'movement_consistency': 0.50,  # Inconsistent
            'base_snr': 33.0,  # Very high SNR (more reflective surface)
        },
        9: {  # Young man, little muscular
            'walking_speed_range': (0.022, 0.035),  # Fast
            'body_scale': 1.15,  # Large build (muscular)
            'movement_consistency': 0.80,  # Good consistency
            'base_snr': 30.0,  # High SNR (muscular, reflective)
        },
    }
    
    return characteristics.get(subject_id, {
        'walking_speed_range': (0.015, 0.025),
        'body_scale': 1.0,
        'movement_consistency': 0.7,
        'base_snr': 24.0,
    })


def map_subject_to_combined_class(original_subject_id):
    """
    Map original 10 subjects to 6 combined classes
    Groups by similar characteristics for better learning
    
    Mapping:
    - Class 0: Young adult, slightly thin (subject 0)
    - Class 1: Woman 30s, little fat (subject 1)
    - Class 2: Old man 60s, active (subjects 2, 3)
    - Class 3: Old woman 60s (subjects 4, 8 - weak and fat)
    - Class 4: Young woman 20s, fit (subjects 5, 6)
    - Class 5: Young man (subjects 7, 9 - muscular and little muscular)
    
    Args:
        original_subject_id: Original subject ID (0-9)
    
    Returns:
        combined_class_id: Combined class ID (0-5)
    """
    mapping = {
        0: 0,  # Young adult, slightly thin
        1: 1,  # Woman 30s, little fat
        2: 2,  # Old man 60s, active
        3: 2,  # Old man 60s, active (combined with 2)
        4: 3,  # Old woman 60s, weak
        5: 4,  # Young woman 20s, fit
        6: 4,  # Young woman 20s, fit (combined with 5)
        7: 5,  # Young man, muscular
        8: 3,  # Old woman, fat (combined with 4 - both old women)
        9: 5,  # Young man, little muscular (combined with 7)
    }
    return mapping[original_subject_id]


def remap_labels_to_combined_classes(y_labels):
    """
    Remap original subject labels (0-9) to combined class labels (0-5)
    Useful when you have existing datasets with original labels
    
    Note: If labels are already in range 0-5, this function will still work
    (it will map them correctly), but it's redundant if generate_dataset
    already returned combined class IDs.
    
    Args:
        y_labels: Array of original subject labels (0-9) or already combined (0-5)
    
    Returns:
        remapped_labels: Array of combined class labels (0-5)
    """
    # Check if labels are already in range 0-5 (already remapped)
    unique_labels = np.unique(y_labels)
    if len(unique_labels) > 0 and max(unique_labels) <= 5 and min(unique_labels) >= 0:
        # Labels are already in range 0-5, but verify they're valid combined classes
        # If they are, return as-is (no need to remap)
        if all(label in [0, 1, 2, 3, 4, 5] for label in unique_labels):
            return y_labels
    
    # Otherwise, remap from original subject IDs (0-9) to combined classes (0-5)
    remapped = np.array([map_subject_to_combined_class(int(label)) for label in y_labels])
    return remapped


def create_temporal_sequence(base_vertices, num_frames=30, num_points=200, 
                            subject_id=None, augment=True, random_seed=None):
    """
    Create synthetic temporal sequence from static point cloud to simulate radar data.
    
    This function transforms a static 3D mesh scan into a temporal sequence that mimics
    mmWave radar data. Key features:
    - Uses FPS for shape-preserving downsampling
    - Applies rigid transformations (rotation, translation) per sequence
    - Generates progressive walking motion across frames
    - Computes velocity and SNR channels based on subject characteristics
    - Applies distance-dependent noise (radar-like behavior)
    
    Following the paper's approach: rigid transformations are applied once per sequence
    (not per frame) to preserve relative positions between points (body shape).
    
    Args:
        base_vertices: (N, 3) original point cloud from PLY file
        num_frames: Number of temporal frames (default: 30, matching radar frame rate)
        num_points: Points per frame (default: 200, typical radar output)
        subject_id: Subject ID (0-9) for subject-specific characteristics
        augment: Whether to apply data augmentation
        random_seed: Random seed for reproducibility
    
    Returns:
        sequence: (30, 200, 5) numpy array - frames × points × channels
                  Channels: (x, y, z, velocity, snr)
        label: subject_id (for tracking)
    """
    # Get subject-specific characteristics
    if subject_id is not None:
        subj_chars = get_subject_characteristics(subject_id)
    else:
        subj_chars = get_subject_characteristics(0)  # Default
    
    # Sample points ONCE for the entire sequence using FPS (preserves body shape)
    # Use deterministic seed based on subject_id to ensure same subject gets similar sampling
    # Save random state before FPS, then restore it for augmentation
    rng_state = None
    if subject_id is not None:
        # Save current random state
        rng_state = np.random.get_state()
        # Deterministic seed per subject - same subject gets similar points
        # This ensures consistency: subject 0 always gets similar point selection
        np.random.seed(subject_id * 1000)  # Different seed per subject
    
    if len(base_vertices) > num_points:
        # Use FPS for better shape preservation (uniform coverage, preserves body proportions)
        # Pass subject_id to make FPS deterministic per subject
        sampled, _ = farthest_point_sampling(base_vertices, num_points, subject_id=subject_id)
    else:
        sampled = base_vertices.copy()
        # Pad if needed
        if len(sampled) < num_points:
            padding = np.zeros((num_points - len(sampled), 3))
            sampled = np.vstack([sampled, padding])
    
    # Restore random state for augmentation (or set random_seed if provided)
    if rng_state is not None:
        np.random.set_state(rng_state)
    if random_seed is not None:
        np.random.seed(random_seed)
        
    # Apply subject-specific body scale (differentiates body sizes)
    if augment:
        sampled = sampled * subj_chars['body_scale']
    
    # Initialize walking parameters (used only if augment=True)
    walk_speed = 0.0
    walk_direction = 0.0
    radar_origin = None
    
    # Apply rigid transformation ONCE per sequence (as in paper)
    if augment:
        # Random rotation: 0 to 360 degrees around z-axis
        angle = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = create_rotation_matrix_z(angle)
        
        # Random translation: 0 to 3 meters along x and y axes only (NOT z)
        translation_xy = np.random.uniform(0, 3.0, size=(1, 2))  # x, y only
        base_translation = np.hstack([translation_xy, np.zeros((1, 1))])  # z = 0
        
        # Apply SAME rigid transformation to all points (preserves body shape)
        sampled = sampled @ rotation_matrix.T + base_translation
        
        # Subject-specific walking speed (based on fitness/age)
        walk_speed = np.random.uniform(*subj_chars['walking_speed_range'])
        walk_direction = np.random.uniform(0, 2 * np.pi)  # Random walking direction
        
        # Set radar origin (simulate wall-mounted radar at room corner, 1.5m height)
        radar_origin = np.array([0.0, 0.0, 1.5])  # (x, y, z) in meters
    
    # Now create temporal sequence with radar-like characteristics
    sequence = []
    prev_frame_xyz = None
    
    for frame_idx in range(num_frames):
        frame_xyz = sampled.copy()
        
        if augment:
            # 1. Progressive translation (simulating walking through space)
            # Subject-specific walking speed
            walk_translation = np.array([
                [np.cos(walk_direction) * walk_speed * frame_idx,
                 np.sin(walk_direction) * walk_speed * frame_idx,
                 0]  # No vertical movement
            ])
            frame_xyz = frame_xyz + walk_translation
            
            # 2. Transform relative to radar origin (radar coordinate system)
            # Radar data is relative to antenna position
            frame_xyz = frame_xyz - radar_origin
            
            # 3. Distance-dependent noise (radar signal degradation with distance)
            # Farther points = more noise (typical radar behavior)
            distances = np.linalg.norm(frame_xyz, axis=1, keepdims=True)  # (200, 1)
            # Base noise modulated by movement consistency (fit = less noise, weak = more noise)
            base_noise = 0.0005 / subj_chars['movement_consistency']
            noise_std = base_noise + 0.0005 * (distances / 5.0)  # More noise for far points
            noise = np.random.normal(0, noise_std, frame_xyz.shape)
            frame_xyz = frame_xyz + noise
            
            # 4. Compute velocity (magnitude) from frame-to-frame change
            if prev_frame_xyz is not None:
                # Velocity = change in position per frame
                velocity_vectors = frame_xyz - prev_frame_xyz  # (200, 3)
                velocity_magnitude = np.linalg.norm(velocity_vectors, axis=1)  # (200,)
                # Add subject-specific velocity variation (fit = smoother, weak = more erratic)
                if subj_chars['movement_consistency'] < 0.7:
                    # Less consistent subjects: more velocity variation
                    velocity_noise = np.random.normal(0, 0.002, velocity_magnitude.shape)
                    velocity_magnitude = np.abs(velocity_magnitude + velocity_noise)
            else:
                velocity_magnitude = np.zeros(num_points)  # First frame: zero velocity
            
            # 5. Compute SNR (Signal-to-Noise Ratio) based on distance and subject
            distances_1d = np.linalg.norm(frame_xyz, axis=1)  # (200,)
            # Subject-specific base SNR (larger bodies = higher SNR, more reflective)
            # SNR decreases ~2dB per meter (typical radar path loss)
            distance_loss = -2.0 * distances_1d
            # Add random variation (sensor noise, multipath, etc.)
            snr_noise = np.random.normal(0, 1.5, distances_1d.shape)
            snr = subj_chars['base_snr'] + distance_loss + snr_noise
            # Clamp to realistic range (0-40 dB)
            snr = np.clip(snr, 0.0, 40.0)
        else:
            # No augmentation: zero velocity, constant SNR
            velocity_magnitude = np.zeros(num_points)
            snr = np.full(num_points, 20.0)  # Constant SNR
        
        # Normalize channels to similar scales for effective multi-channel learning
        # Without normalization, velocity and SNR would be ignored due to scale mismatch
        
        # Velocity normalization: scale to match coordinate variations
        # Typical velocity: 0.01-0.03 m/frame → scale to 0.1-0.3 range
        velocity_magnitude = velocity_magnitude * 10.0
        
        # SNR normalization: center and scale to match coordinate range
        # Typical SNR: 0-40 dB → center at 20, scale to [-2, 2] range
        snr = (snr - 20.0) / 10.0
        
        # Combine: (x, y, z, velocity, snr) = 5 channels
        frame_data = np.hstack([
            frame_xyz,  # (200, 3) - keep as-is
            velocity_magnitude.reshape(-1, 1),  # (200, 1) - now normalized
            snr.reshape(-1, 1)  # (200, 1) - now normalized
        ])  # (200, 5)
        
        sequence.append(frame_data)
        prev_frame_xyz = frame_xyz.copy()
    
    return np.array(sequence), subject_id  # (30, 200, 5), label


def generate_dataset(scan_indices, scans_dir, num_sequences_per_scan=10, 
                    num_frames=30, num_points=200, verbose=True):
    """
    Generate sequences from scans
    
    Args:
        scan_indices: List of scan indices to process
        scans_dir: Directory containing PLY files
        num_sequences_per_scan: Number of sequences to generate per scan
        num_frames: Frames per sequence (30)
        num_points: Points per frame (200)
        verbose: Whether to print progress
    
    Returns:
        X: (N, 30, 200, 5) sequences - channels: (x, y, z, velocity, snr)
        y: (N,) labels (already mapped to combined classes 0-5)
    """
    X = []
    y = []
    missing_scans = []
    error_scans = []
    subject_counts = {}  # Track sequences per subject
    
    for idx, scan_idx in enumerate(scan_indices):
        if verbose and (idx + 1) % 10 == 0:
            print(f"  Processing scan {idx+1}/{len(scan_indices)}...")
        
        # Load scan
        ply_path = os.path.join(scans_dir, f'tr_scan_{scan_idx:03d}.ply')
        if not os.path.exists(ply_path):
            missing_scans.append(scan_idx)
            if verbose:
                print(f"  Warning: {ply_path} not found, skipping")
            continue
            
        try:
            vertices = load_ply_file(ply_path)
            original_subject_id = scan_idx // 10
            combined_class_id = map_subject_to_combined_class(original_subject_id)  # Map to combined class
            
            # Track subject counts
            if original_subject_id not in subject_counts:
                subject_counts[original_subject_id] = 0
            
            # Generate multiple sequences per scan (augmentation)
            sequences_generated = 0
            for seq_idx in range(num_sequences_per_scan):
                try:
                    sequence, _ = create_temporal_sequence(
                        vertices, 
                        num_frames=num_frames,
                        num_points=num_points,
                        subject_id=original_subject_id,  # Still use original for characteristics
                        augment=True,
                        random_seed=scan_idx * 1000 + seq_idx  # Reproducible
                    )
                    X.append(sequence)
                    y.append(combined_class_id)  # Use combined class ID for label
                    subject_counts[original_subject_id] += 1
                    sequences_generated += 1
                except Exception as seq_e:
                    # Error generating a single sequence
                    if verbose and original_subject_id in [7, 9]:  # Debug subjects 7 and 9
                        print(f"    Error generating sequence {seq_idx} for scan {scan_idx} (subject {original_subject_id}): {seq_e}")
                    error_scans.append((scan_idx, f"seq_{seq_idx}: {str(seq_e)}"))
                    continue
                
        except Exception as e:
            error_scans.append((scan_idx, str(e)))
            if verbose:
                subject_id = scan_idx // 10
                print(f"  Error processing scan {scan_idx} (subject {subject_id}): {e}")
                if subject_id in [7, 9]:  # Extra debugging for subjects 7 and 9
                    import traceback
                    print(f"    Full traceback for subject {subject_id}:")
                    traceback.print_exc()
            continue
    
    # Report statistics
    if verbose:
        print(f"\n  Generated {len(X)} sequences total")
        if missing_scans:
            missing_subjects = sorted(set([s // 10 for s in missing_scans]))
            print(f"  ⚠️  {len(missing_scans)} scans missing (subjects: {missing_subjects})")
        if error_scans:
            error_subjects = sorted(set([s[0] // 10 for s in error_scans]))
            print(f"  ⚠️  {len(error_scans)} scans had errors (subjects: {error_subjects})")
            # Show first few errors
            for scan_idx, error_msg in error_scans[:3]:
                print(f"      Scan {scan_idx}: {error_msg}")
        
        # Show subject distribution
        print(f"\n  Sequences per original subject:")
        for subj_id in sorted(subject_counts.keys()):
            combined_class = map_subject_to_combined_class(subj_id)
            print(f"    Subject {subj_id} → Class {combined_class}: {subject_counts[subj_id]} sequences")
    
    return np.array(X), np.array(y)
