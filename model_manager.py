# model_manager.py
"""
ModelManager adapter for tiered_fl_system.
This is used to test the orchestrator and FL workflow with a simple trajectory prediction model.

Each FL client must set the environment variable DATA_CSV_PATH to the path
of its local trajectory CSV (columns: id, lon, lat, time).

Optional env vars:
  LOCAL_EPOCHS   — epochs per FL round (default: 3)
  SEQ_LENGTH     — sliding-window sequence length (default: 20)
  VAL_SPLIT      — fraction of trajectories held out for evaluation (default: 0.2)
"""

import os
import numpy as np
import torch

from trajectory_predictor import TrajectoryPredictor, TrajectoryDataset
from data_loader import load_trajectories, split_train_test


class ModelManager:

    def __init__(self):
        seq_len = int(os.environ.get("SEQ_LENGTH", 20))
        self.local_epochs = int(os.environ.get("LOCAL_EPOCHS", 3))
        self.val_split = float(os.environ.get("VAL_SPLIT", 0.2))

        self.predictor = TrajectoryPredictor(sequence_length=seq_len)

        # Load data only if a CSV path is provided (clients set this;
        # the orchestrator instantiates ModelManager without data to seed
        # the initial global parameters).
        csv_path = os.environ.get("DATA_CSV_PATH", "")
        if csv_path and os.path.exists(csv_path):
            trajectories, _ = load_trajectories(csv_path)
            self.train_trajectories, self.eval_trajectories = split_train_test(
                trajectories, test_ratio=self.val_split, random_seed=42
            )
            # Fit the scaler so the predictor is in a usable state
            self.predictor.prepare_data(self.train_trajectories)
            self.predictor.is_fitted = True
        else:
            self.train_trajectories = []
            self.eval_trajectories = []

    # ------------------------------------------------------------------
    # FL contract
    # ------------------------------------------------------------------

    def get_model(self):
        return self.predictor.model

    def get_model_parameters(self):
        return [
            val.cpu().numpy()
            for val in self.predictor.model.state_dict().values()
        ]

    def set_model_parameters(self, parameters):
        state_dict = self.predictor.model.state_dict()
        new_state = {
            key: torch.tensor(param)
            for key, param in zip(state_dict.keys(), parameters)
        }
        self.predictor.model.load_state_dict(new_state)

    def fit_model(self):
        if not self.train_trajectories:
            raise RuntimeError(
                "No training data. Set DATA_CSV_PATH before starting the client."
            )
        self.predictor.train(
            self.train_trajectories,
            epochs=self.local_epochs,
            val_split=0.0,  # use all local data for training each round
        )
        # Return total sliding-window samples used
        seq_len = self.predictor.sequence_length
        return sum(
            max(0, len(t) - seq_len) for t in self.train_trajectories
        )

    def evaluate_model(self):
        if not self.eval_trajectories:
            return 0.0, 0.0, 0

        seq_len = self.predictor.sequence_length
        criterion = torch.nn.MSELoss()
        self.predictor.model.eval()

        total_loss = 0.0
        n_samples = 0

        dataloader = self.predictor.prepare_data(
            self.eval_trajectories, batch_size=32, shuffle=False
        )

        with torch.no_grad():
            for sequences, targets in dataloader:
                sequences = sequences.to(self.predictor.device)
                targets = targets.to(self.predictor.device)
                preds = self.predictor.model(sequences)
                total_loss += criterion(preds, targets).item() * len(sequences)
                n_samples += len(sequences)

        loss = total_loss / n_samples if n_samples > 0 else 0.0
        # Accuracy is not meaningful for regression; return 0.0
        return loss, 0.0, n_samples