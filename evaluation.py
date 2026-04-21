import numpy as np
from trajectory_predictor import haversine_distance


def evaluate_model(predictor, test_trajectories, sequence_length=20):
    """
    Comprehensive evaluation of the trajectory prediction model
    
    Args:
        predictor: Trained TrajectoryPredictor object
        test_trajectories: List of test trajectory arrays
        sequence_length: Number of past points used for prediction
        
    Returns:
        Dictionary with evaluation metrics
    """
    errors = []
    predictions = []
    actuals = []
    
    print("\n" + "="*70)
    print("EVALUATING MODEL")
    print("="*70)
    
    # Evaluate on all test samples
    n_evaluated = 0
    n_skipped = 0
    
    for traj_idx, trajectory in enumerate(test_trajectories):
        if len(trajectory) <= sequence_length:
            n_skipped += 1
            continue
        
        # Test on all possible windows in this trajectory
        for i in range(len(trajectory) - sequence_length):
            past_points = trajectory[i:i+sequence_length]
            actual_next = trajectory[i+sequence_length]
            
            try:
                predicted_next = predictor.predict(past_points)
                
                # Calculate error in meters
                error = haversine_distance(
                    actual_next[0], actual_next[1],
                    predicted_next[0], predicted_next[1]
                )
                
                errors.append(error)
                predictions.append(predicted_next)
                actuals.append(actual_next)
                n_evaluated += 1
                
            except Exception as e:
                print(f"Warning: Prediction failed for trajectory {traj_idx}, position {i}: {e}")
                continue
    
    print(f"Evaluated: {n_evaluated} predictions")
    print(f"Skipped: {n_skipped} trajectories (too short)")
    
    if len(errors) == 0:
        print("ERROR: No predictions to evaluate!")
        return None
    
    errors = np.array(errors)
    
    # Calculate metrics
    metrics = {
        'n_predictions': len(errors),
        'mean_error_m': np.mean(errors),
        'median_error_m': np.median(errors),
        'std_error_m': np.std(errors),
        'min_error_m': np.min(errors),
        'max_error_m': np.max(errors),
        'percentile_50_m': np.percentile(errors, 50),
        'percentile_75_m': np.percentile(errors, 75),
        'percentile_90_m': np.percentile(errors, 90),
        'percentile_95_m': np.percentile(errors, 95),
        'rmse_m': np.sqrt(np.mean(errors ** 2)),
    }
    
    # Print results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Number of predictions: {metrics['n_predictions']}")
    print(f"\nDistance Errors (meters):")
    print(f"  Mean:       {metrics['mean_error_m']:.2f} m")
    print(f"  Median:     {metrics['median_error_m']:.2f} m")
    print(f"  Std Dev:    {metrics['std_error_m']:.2f} m")
    print(f"  Min:        {metrics['min_error_m']:.2f} m")
    print(f"  Max:        {metrics['max_error_m']:.2f} m")
    print(f"  RMSE:       {metrics['rmse_m']:.2f} m")
    print(f"\nPercentiles:")
    print(f"  50th (median): {metrics['percentile_50_m']:.2f} m")
    print(f"  75th:          {metrics['percentile_75_m']:.2f} m")
    print(f"  90th:          {metrics['percentile_90_m']:.2f} m")
    print(f"  95th:          {metrics['percentile_95_m']:.2f} m")
    print("="*70)
    
    return metrics


def show_example_predictions(predictor, test_trajectories, n_examples=3, sequence_length=20):
    """
    Show example predictions with actual vs predicted coordinates
    
    Args:
        predictor: Trained TrajectoryPredictor
        test_trajectories: List of test trajectories
        n_examples: Number of examples to show
        sequence_length: Sequence length used for prediction
    """
    print("\n" + "="*70)
    print("EXAMPLE PREDICTIONS")
    print("="*70)
    
    examples_shown = 0
    
    for traj_idx, trajectory in enumerate(test_trajectories):
        if examples_shown >= n_examples:
            break
            
        if len(trajectory) <= sequence_length:
            continue
        
        # Use the first valid window from this trajectory
        past_points = trajectory[:sequence_length]
        actual_next = trajectory[sequence_length]
        
        try:
            predicted_next = predictor.predict(past_points)
            
            error = haversine_distance(
                actual_next[0], actual_next[1],
                predicted_next[0], predicted_next[1]
            )
            
            print(f"\nExample {examples_shown + 1}:")
            print(f"  Trajectory: {traj_idx + 1}")
            print(f"  Input: {sequence_length} past points")
            print(f"  Actual next:    (lat={actual_next[0]:.6f}, lon={actual_next[1]:.6f})")
            print(f"  Predicted next: (lat={predicted_next[0]:.6f}, lon={predicted_next[1]:.6f})")
            print(f"  Error: {error:.2f} meters")
            
            examples_shown += 1
            
        except Exception as e:
            continue
    
    print("="*70)


def evaluate_by_trajectory_length(predictor, test_trajectories, sequence_length=20):
    """
    Evaluate model performance grouped by trajectory length
    
    Args:
        predictor: Trained TrajectoryPredictor
        test_trajectories: List of test trajectories
        sequence_length: Sequence length for prediction
    """
    print("\n" + "="*70)
    print("PERFORMANCE BY TRAJECTORY LENGTH")
    print("="*70)
    
    # Group by trajectory length ranges
    length_ranges = [
        (21, 30, "Short (21-30)"),
        (31, 50, "Medium (31-50)"),
        (51, 100, "Long (51-100)"),
        (101, float('inf'), "Very Long (>100)")
    ]
    
    for min_len, max_len, label in length_ranges:
        errors = []
        
        for trajectory in test_trajectories:
            traj_len = len(trajectory)
            
            if min_len <= traj_len < max_len:
                # Evaluate on this trajectory
                for i in range(len(trajectory) - sequence_length):
                    past = trajectory[i:i+sequence_length]
                    actual = trajectory[i+sequence_length]
                    
                    try:
                        predicted = predictor.predict(past)
                        error = haversine_distance(
                            actual[0], actual[1],
                            predicted[0], predicted[1]
                        )
                        errors.append(error)
                    except:
                        continue
        
        if len(errors) > 0:
            errors = np.array(errors)
            print(f"\n{label}:")
            print(f"  Predictions: {len(errors)}")
            print(f"  Mean error: {np.mean(errors):.2f} m")
            print(f"  Median error: {np.median(errors):.2f} m")
            print(f"  90th percentile: {np.percentile(errors, 90):.2f} m")
        else:
            print(f"\n{label}: No trajectories in this range")
    
    print("="*70)


def multi_step_evaluation(predictor, test_trajectories, n_steps=5, sequence_length=20):
    """
    Evaluate multi-step prediction (predicting multiple future locations)
    
    Args:
        predictor: Trained TrajectoryPredictor
        test_trajectories: List of test trajectories
        n_steps: Number of future steps to predict
        sequence_length: Sequence length for initial prediction
    """
    print("\n" + "="*70)
    print(f"MULTI-STEP PREDICTION EVALUATION (predicting {n_steps} steps ahead)")
    print("="*70)
    
    step_errors = [[] for _ in range(n_steps)]
    
    for trajectory in test_trajectories:
        if len(trajectory) <= sequence_length + n_steps:
            continue
        
        # Start from the beginning
        current_seq = trajectory[:sequence_length].copy()
        
        for step in range(n_steps):
            actual = trajectory[sequence_length + step]
            
            try:
                predicted = predictor.predict(current_seq)
                error = haversine_distance(
                    actual[0], actual[1],
                    predicted[0], predicted[1]
                )
                step_errors[step].append(error)
                
                # Update sequence for next prediction
                current_seq = np.vstack([current_seq[1:], predicted])
                
            except:
                break
    
    # Print results for each step
    print(f"\nError accumulation over {n_steps} future steps:")
    for step in range(n_steps):
        if len(step_errors[step]) > 0:
            errors = np.array(step_errors[step])
            print(f"  Step {step+1}: {np.mean(errors):.2f} m (±{np.std(errors):.2f})")
        else:
            print(f"  Step {step+1}: No data")
    
    print("="*70)


if __name__ == "__main__":
    print("Evaluation Module")
    print("Import this module to use: from evaluation import evaluate_model")
