"""
Main Training Script - Complete Workflow

This script provides a complete workflow:
1. Load your data (CSV format: id, lon, lat, time)
2. Analyze dataset
3. Train model with automatic parameter selection
4. Evaluate performance
5. Save trained model

Usage:
    python train.py path/to/your/data.csv [path/to/save/model.pth]
"""

import sys
import numpy as np
from trajectory_predictor import TrajectoryPredictor
from data_loader import load_trajectories, analyze_dataset, recommend_parameters, split_train_test
from evaluation import evaluate_model, show_example_predictions


def train_model(csv_filepath, sequence_length=20, test_ratio=0.2, save_path='trained_model.pth'):
    """
    Complete training workflow
    
    Args:
        csv_filepath: Path to CSV file with columns: id, lon, lat, time
        sequence_length: Number of past points to use (default: 20)
        test_ratio: Fraction of data for testing (default: 0.2)
        save_path: Where to save trained model (default: 'trained_model.pth')
    
    Returns:
        Trained TrajectoryPredictor object
    """
    
    print("="*70)
    print("TRAJECTORY NEXT LOCATION PREDICTION - TRAINING WORKFLOW")
    print("="*70)
    
    # ========================================================================
    # STEP 1: LOAD DATA
    # ========================================================================
    print("\n[STEP 1/6] Loading trajectories from CSV...")
    print("-"*70)
    
    trajectories, trajectory_info = load_trajectories(csv_filepath)
    trajectories = trajectories[:100]  # Limit for faster testing; remove in real use
    
    if len(trajectories) == 0:
        print("ERROR: No trajectories loaded!")
        return None
    
    # ========================================================================
    # STEP 2: ANALYZE DATASET
    # ========================================================================
    print("\n[STEP 2/6] Analyzing dataset...")
    print("-"*70)
    
    stats = analyze_dataset(trajectories, sequence_length)
    
    if stats['n_usable'] < 2:
        print("\nERROR: Not enough usable trajectories!")
        print(f"Need at least 2 trajectories with >{sequence_length} points")
        print(f"Currently have: {stats['n_usable']}")
        print("\nSuggestions:")
        print(f"  1. Reduce sequence_length (current: {sequence_length})")
        print(f"  2. Collect more data per trajectory")
        return None
    
    # ========================================================================
    # STEP 3: GET RECOMMENDED PARAMETERS
    # ========================================================================
    print("\n[STEP 3/6] Getting recommended training parameters...")
    print("-"*70)
    
    config = recommend_parameters(stats)
    config["epochs"] = 10  # Override for faster model with no drop in accuracy
    
    # ========================================================================
    # STEP 4: SPLIT DATA
    # ========================================================================
    print("\n[STEP 4/6] Splitting into train/test sets...")
    print("-"*70)
    
    train_trajectories, test_trajectories = split_train_test(
        trajectories, 
        test_ratio=test_ratio,
        random_seed=42
    )
    
    # ========================================================================
    # STEP 5: TRAIN MODEL
    # ========================================================================
    print("\n[STEP 5/6] Training model...")
    print("-"*70)
    
    predictor = TrajectoryPredictor(
        sequence_length=sequence_length,
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers']
    )
    
    print(f"\nStarting training with:")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Model dimension: {config['d_model']}")
    print(f"  Attention heads: {config['nhead']}")
    print(f"  Layers: {config['num_layers']}")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Learning rate: {config['lr']}")
    print()
    
    try:
        predictor.train(
            train_trajectories,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            lr=config['lr'],
            val_split=0.2
        )
    except Exception as e:
        print(f"\nERROR during training: {e}")
        return None
    
    # ========================================================================
    # STEP 6: EVALUATE MODEL
    # ========================================================================
    print("\n[STEP 6/6] Evaluating model on test set...")
    print("-"*70)
    
    metrics = evaluate_model(predictor, test_trajectories, sequence_length)
    
    if metrics is None:
        print("WARNING: Could not evaluate model")
    else:
        # Show example predictions
        show_example_predictions(predictor, test_trajectories, n_examples=3, sequence_length=sequence_length)
    
    # ========================================================================
    # SAVE MODEL
    # ========================================================================
    print("\n" + "="*70)
    print("SAVING MODEL")
    print("="*70)
    
    predictor.save(save_path)
    print(f"✓ Model saved to: {save_path}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nModel Summary:")
    print(f"  Trained on: {len(train_trajectories)} trajectories")
    print(f"  Tested on: {len(test_trajectories)} trajectories")
    print(f"  Sequence length: {sequence_length} points")
    
    if metrics:
        print(f"\nPerformance:")
        print(f"  Mean error: {metrics['mean_error_m']:.2f} meters")
        print(f"  Median error: {metrics['median_error_m']:.2f} meters")
        print(f"  90th percentile: {metrics['percentile_90_m']:.2f} meters")
    
    print(f"\nModel saved to: {save_path}")
    print("\nTo use the model:")
    print("  from trajectory_predictor import TrajectoryPredictor")
    print("  predictor = TrajectoryPredictor()")
    print(f"  predictor.load('{save_path}')")
    print("  next_location = predictor.predict(past_20_points)")
    
    print("="*70)
    
    return predictor


def quick_train(csv_filepath, save_path='trained_model.pth'):
    """
    Quick training with default parameters
    
    Usage:
        python train.py your_data.csv
    """
    return train_model(
        csv_filepath,
        sequence_length=20,
        test_ratio=0.2,
        save_path=save_path
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train.py path/to/your/data.csv [path/to/save/model.pth]")
        print("\nExample:")
        print("  python train.py trajectories.csv")
        print("  python train.py trajectories.csv models/my_model.pth")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    save_path = sys.argv[2] if len(sys.argv) > 2 else 'trained_model.pth'
    
    # Check if file exists
    import os
    if not os.path.exists(csv_path):
        print(f"ERROR: File not found: {csv_path}")
        sys.exit(1)
    
    # Train model
    predictor = quick_train(csv_path, save_path=save_path)
    
    if predictor is None:
        print("\nTraining failed. Please check the errors above.")
        sys.exit(1)
