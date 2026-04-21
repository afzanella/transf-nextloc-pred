"""
CSV format: id, lon, lat, time

Handles variable-length trajectories automatically.
"""

import pandas as pd
import numpy as np


def load_trajectories(csv_filepath):
    """
    Load trajectories from CSV
    
    Expected columns: id, lon, lat, time
    
    Returns:
        List of numpy arrays, each shape (n_points, 2) with [lat, lon]
        
    Note: Automatically handles trajectories of different lengths!
    """
    # Read CSV
    df = pd.read_csv(csv_filepath)
    
    # Validate columns
    required_cols = ['id', 'lon', 'lat', 'time']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV must have columns: {required_cols}. Found: {df.columns.tolist()}")
    
    print(f"Loaded CSV with {len(df)} data points")
    print(f"Columns: {df.columns.tolist()}")
    
    # Group by car/user ID and create trajectories
    trajectories = []
    trajectory_info = []
    
    for car_id, group in df.groupby('id'):
        # Sort by time to ensure correct sequence
        group = group.sort_values('time')
        
        # Extract [lat, lon] pairs (note: we swap lon, lat to lat, lon)
        trajectory = group[['lat', 'lon']].values
        
        trajectories.append(trajectory)
        trajectory_info.append({
            'id': car_id,
            'length': len(trajectory),
            'time_start': group['time'].min(),
            'time_end': group['time'].max()
        })
    
    # Print summary
    print(f"\nTrajectories loaded: {len(trajectories)}")
    print(f"Trajectory lengths:")
    for info in trajectory_info:
        print(f"  {info['id']}: {info['length']} points (time: {info['time_start']:.1f} - {info['time_end']:.1f})")
    
    return trajectories, trajectory_info


def analyze_dataset(trajectories, sequence_length=20):
    """
    Analyze trajectory dataset to understand its characteristics
    
    Args:
        trajectories: List of numpy arrays
        sequence_length: Required sequence length for training
        
    Returns:
        Dictionary with dataset statistics
    """
    lengths = [len(t) for t in trajectories]
    
    # Calculate usable trajectories
    usable = sum(1 for l in lengths if l > sequence_length)
    unusable = len(lengths) - usable
    
    # Calculate total training samples
    total_samples = sum(max(0, l - sequence_length) for l in lengths)
    
    stats = {
        'n_trajectories': len(trajectories),
        'n_usable': usable,
        'n_unusable': unusable,
        'min_length': min(lengths) if lengths else 0,
        'max_length': max(lengths) if lengths else 0,
        'mean_length': np.mean(lengths) if lengths else 0,
        'median_length': np.median(lengths) if lengths else 0,
        'total_samples': total_samples,
        'lengths': lengths
    }
    
    print("\n" + "="*70)
    print("DATASET ANALYSIS")
    print("="*70)
    print(f"Total trajectories: {stats['n_trajectories']}")
    print(f"Usable (>{sequence_length} points): {stats['n_usable']}")
    print(f"Too short (≤{sequence_length} points): {stats['n_unusable']}")
    print(f"\nTrajectory lengths:")
    print(f"  Min: {stats['min_length']} points")
    print(f"  Max: {stats['max_length']} points")
    print(f"  Mean: {stats['mean_length']:.1f} points")
    print(f"  Median: {stats['median_length']:.1f} points")
    print(f"\nTotal training samples: {stats['total_samples']}")
    
    # Show length distribution
    print(f"\nLength distribution:")
    ranges = [(0, 20), (21, 30), (31, 50), (51, 100), (101, float('inf'))]
    for low, high in ranges:
        count = sum(1 for l in lengths if low <= l < high)
        if high == float('inf'):
            print(f"  >{low} points: {count} trajectories")
        else:
            print(f"  {low}-{high} points: {count} trajectories")
    
    print("="*70)
    
    return stats


def recommend_parameters(stats):
    """
    Recommend training parameters based on dataset size
    
    Args:
        stats: Dictionary from analyze_dataset()
        
    Returns:
        Dictionary with recommended parameters
    """
    total_samples = stats['total_samples']
    n_trajectories = stats['n_usable']
    
    # Determine model size
    if total_samples < 300 or n_trajectories < 10:
        config = {
            'model_size': 'small',
            'd_model': 64,
            'nhead': 4,
            'num_layers': 2,
            'epochs': 30,
            'batch_size': 8,
            'lr': 0.001
        }
    elif total_samples < 1000:
        config = {
            'model_size': 'medium',
            'd_model': 128,
            'nhead': 8,
            'num_layers': 3,
            'epochs': 50,
            'batch_size': 16,
            'lr': 0.001
        }
    else:
        config = {
            'model_size': 'large',
            'd_model': 128,
            'nhead': 8,
            'num_layers': 4,
            'epochs': 80,
            'batch_size': 32,
            'lr': 0.001
        }
    
    print("\n" + "="*70)
    print("RECOMMENDED TRAINING PARAMETERS")
    print("="*70)
    print(f"Model size: {config['model_size'].upper()}")
    print(f"  d_model: {config['d_model']}")
    print(f"  attention heads: {config['nhead']}")
    print(f"  layers: {config['num_layers']}")
    print(f"  epochs: {config['epochs']}")
    print(f"  batch_size: {config['batch_size']}")
    print(f"  learning rate: {config['lr']}")
    print("="*70)
    
    return config


def split_train_test(trajectories, test_ratio=0.2, random_seed=42):
    """
    Split trajectories into train and test sets
    
    Args:
        trajectories: List of trajectory arrays
        test_ratio: Fraction for test set (default: 0.2)
        random_seed: Random seed for reproducibility
        
    Returns:
        train_trajectories, test_trajectories
    """
    np.random.seed(random_seed)
    indices = np.random.permutation(len(trajectories))
    
    n_test = int(len(trajectories) * test_ratio)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    train_trajectories = [trajectories[i] for i in train_indices]
    test_trajectories = [trajectories[i] for i in test_indices]
    
    print(f"\nData split:")
    print(f"  Training: {len(train_trajectories)} trajectories")
    print(f"  Test: {len(test_trajectories)} trajectories")
    
    return train_trajectories, test_trajectories


if __name__ == "__main__":
    print("Data Loader Module")
    print("Import this module to use: from data_loader import load_trajectories")
