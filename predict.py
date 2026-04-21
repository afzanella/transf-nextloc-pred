import numpy as np
import sys
from trajectory_predictor import TrajectoryPredictor, haversine_distance
from data_loader import load_trajectories


def predict_next_location(predictor, past_points):
    """
    Predict next location given past points
    
    Args:
        predictor: Trained TrajectoryPredictor
        past_points: numpy array of shape (sequence_length, 2) with [lat, lon]
    
    Returns:
        (lat, lon) tuple of predicted next location
    """
    prediction = predictor.predict(past_points)
    return tuple(prediction)


def predict_multiple_steps(predictor, initial_points, n_steps=5):
    """
    Predict multiple future locations iteratively
    
    Args:
        predictor: Trained TrajectoryPredictor
        initial_points: numpy array of shape (sequence_length, 2)
        n_steps: number of future steps to predict
    
    Returns:
        List of predicted (lat, lon) tuples
    """
    predictions = []
    current_sequence = initial_points.copy()
    
    for step in range(n_steps):
        # Predict next point
        next_point = predictor.predict(current_sequence)
        predictions.append(tuple(next_point))
        
        # Update sequence: remove first point, add predicted point
        current_sequence = np.vstack([current_sequence[1:], next_point])
    
    return predictions


def demo_predictions(model_path='trained_model.pth', csv_path=None):
    """
    Demonstration of how to use the trained model
    
    Args:
        model_path: Path to saved model
        csv_path: Optional path to CSV with trajectory data
    """
    print("="*70)
    print("TRAJECTORY PREDICTION - DEMONSTRATION")
    print("="*70)
    
    # Load model
    print(f"\n[1/3] Loading trained model from {model_path}...")
    predictor = TrajectoryPredictor()
    try:
        predictor.load(model_path)
        print("✓ Model loaded successfully")
        print(f"  Sequence length: {predictor.sequence_length}")
    except Exception as e:
        print(f"ERROR: Could not load model: {e}")
        return
    
    # Load data for demonstration
    if csv_path:
        print(f"\n[2/3] Loading data from {csv_path}...")
        trajectories, _ = load_trajectories(csv_path)
        
        # Find a trajectory with enough points
        demo_traj = None
        for traj in trajectories:
            if len(traj) > predictor.sequence_length + 5:
                demo_traj = traj
                break
        
        if demo_traj is None:
            print("ERROR: No trajectory with enough points for demonstration")
            return
    else:
        # Create synthetic demo data
        print("\n[2/3] Creating synthetic demo data...")
        demo_traj = np.array([
            [40.7128 + i*0.001, -74.0060 + i*0.0005] 
            for i in range(30)
        ])
    
    # Make predictions
    print(f"\n[3/3] Making predictions...")
    print("-"*70)
    
    past_points = demo_traj[:predictor.sequence_length]
    
    # Single-step prediction
    print("\n1. SINGLE-STEP PREDICTION")
    print("-"*70)
    predicted = predict_next_location(predictor, past_points)
    
    print(f"Input: {predictor.sequence_length} past points")
    print(f"Last 3 points:")
    for i in range(-3, 0):
        print(f"  Point {predictor.sequence_length + i + 1}: (lat={past_points[i][0]:.6f}, lon={past_points[i][1]:.6f})")
    
    print(f"\nPredicted next location:")
    print(f"  (lat={predicted[0]:.6f}, lon={predicted[1]:.6f})")
    
    if len(demo_traj) > predictor.sequence_length:
        actual = demo_traj[predictor.sequence_length]
        error = haversine_distance(actual[0], actual[1], predicted[0], predicted[1])
        print(f"\nActual next location:")
        print(f"  (lat={actual[0]:.6f}, lon={actual[1]:.6f})")
        print(f"\nPrediction error: {error:.2f} meters")
    
    # Multi-step prediction
    print("\n2. MULTI-STEP PREDICTION (next 5 locations)")
    print("-"*70)
    multi_predictions = predict_multiple_steps(predictor, past_points, n_steps=5)
    
    for step, pred in enumerate(multi_predictions, 1):
        print(f"  Step {step}: (lat={pred[0]:.6f}, lon={pred[1]:.6f})")
        
        if len(demo_traj) > predictor.sequence_length + step - 1:
            actual = demo_traj[predictor.sequence_length + step - 1]
            error = haversine_distance(actual[0], actual[1], pred[0], pred[1])
            print(f"           Error: {error:.2f} meters")
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETED")
    print("="*70)
    print("\nTo use in your own code:")
    print("""
from trajectory_predictor import TrajectoryPredictor
import numpy as np

# Load model
predictor = TrajectoryPredictor()
predictor.load('trained_model.pth')

# Prepare your past 20 points
past_points = np.array([
    [lat1, lon1],
    [lat2, lon2],
    # ... 18 more points
])

# Predict
next_location = predictor.predict(past_points)
print(f"Next location: {next_location}")
""")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments - use default model
        demo_predictions()
    elif len(sys.argv) == 2:
        # Model path provided
        demo_predictions(model_path=sys.argv[1])
    elif len(sys.argv) == 3:
        # Model path and CSV provided
        demo_predictions(model_path=sys.argv[1], csv_path=sys.argv[2])
    else:
        print("Usage:")
        print("  python predict.py                                    # Use default model")
        print("  python predict.py model.pth                          # Use specific model")
        print("  python predict.py model.pth data.csv                 # Use model with your data")
