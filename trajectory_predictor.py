"""
Trajectory Prediction using Transformers
Predicts next location based on past N lat/lon points

This model automatically handles variable-length trajectories.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class TrajectoryDataset(Dataset):
    """
    Dataset for trajectory sequences
    
    Automatically handles variable-length trajectories by extracting
    fixed-size windows from each trajectory.
    """
    
    def __init__(self, trajectories, sequence_length=20):
        """
        Args:
            trajectories: List of user trajectories, each trajectory is Nx2 array (lat, lon)
            sequence_length: Number of past points to use for prediction
        """
        self.sequence_length = sequence_length
        self.sequences = []
        self.targets = []
        
        # Extract sequences and targets from trajectories
        # This automatically handles variable-length trajectories
        for trajectory in trajectories:
            # Skip trajectories that are too short
            if len(trajectory) <= sequence_length:
                continue
                
            # Create sliding windows from this trajectory
            for i in range(len(trajectory) - sequence_length):
                seq = trajectory[i:i + sequence_length]
                target = trajectory[i + sequence_length]
                self.sequences.append(seq)
                self.targets.append(target)
        
        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor(self.targets[idx])


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TrajectoryTransformer(nn.Module):
    """Transformer-based trajectory prediction model"""
    
    def __init__(self, input_dim=2, d_model=128, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, input_dim)
        )
        
        self.d_model = d_model
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, 2)
        x = self.input_embedding(x) * np.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        
        # Use the last position for prediction
        x = x[:, -1, :]
        output = self.fc(x)
        return output


class TrajectoryPredictor:
    """Main class for training and predicting trajectories"""
    
    def __init__(self, sequence_length=20, d_model=128, nhead=8, num_layers=4, device=None):
        self.sequence_length = sequence_length
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = TrajectoryTransformer(
            input_dim=2,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers
        ).to(self.device)
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def prepare_data(self, trajectories, batch_size=32, shuffle=True):
        """Prepare data for training - handles variable-length trajectories"""
        # Flatten all trajectories for fitting scaler
        all_points = np.vstack([traj for traj in trajectories if len(traj) > 0])
        self.scaler.fit(all_points)
        
        # Normalize trajectories
        normalized_trajectories = [
            self.scaler.transform(traj) for traj in trajectories if len(traj) > self.sequence_length
        ]
        
        # Create dataset and dataloader
        # The dataset automatically handles variable lengths
        dataset = TrajectoryDataset(normalized_trajectories, self.sequence_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
        
        return dataloader
    
    def train(self, trajectories, epochs=50, batch_size=32, lr=0.001, val_split=0.2):
        """Train the model on variable-length trajectories"""
        print(f"Training on device: {self.device}")
        
        # Check data
        usable_trajs = sum(1 for t in trajectories if len(t) > self.sequence_length)
        print(f"Total trajectories: {len(trajectories)}")
        print(f"Usable trajectories (>{self.sequence_length} points): {usable_trajs}")
        
        if usable_trajs < 2:
            raise ValueError(f"Need at least 2 trajectories with >{self.sequence_length} points for training")
        
        # Split data
        n_val = int(len(trajectories) * val_split)
        val_trajectories = trajectories[:n_val] if n_val > 0 else []
        train_trajectories = trajectories[n_val:]
        
        train_loader = self.prepare_data(train_trajectories, batch_size, shuffle=True)
        
        if len(val_trajectories) > 0:
            val_loader = self.prepare_data(val_trajectories, batch_size, shuffle=False)
        else:
            val_loader = None
        
        print(f"Training samples: {len(train_loader.dataset)}")
        if val_loader:
            print(f"Validation samples: {len(val_loader.dataset)}")
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for sequences, targets in train_loader:
                sequences, targets = sequences.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(sequences)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            val_loss = 0
            if val_loader:
                self.model.eval()
                with torch.no_grad():
                    for sequences, targets in val_loader:
                        sequences, targets = sequences.to(self.device), targets.to(self.device)
                        outputs = self.model(sequences)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
            
            if (epoch + 1) % 5 == 0:
                if val_loader:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}")
        
        self.is_fitted = True
        if val_loader:
            print(f"Training completed! Best validation loss: {best_val_loss:.6f}")
        else:
            print(f"Training completed! Final training loss: {train_loss:.6f}")
        
    def predict(self, past_points):
        """
        Predict next location given past points
        
        Args:
            past_points: numpy array of shape (sequence_length, 2) with lat/lon coordinates
            
        Returns:
            Predicted next location as (lat, lon)
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        if len(past_points) != self.sequence_length:
            raise ValueError(f"Expected {self.sequence_length} points, got {len(past_points)}")
        
        # Normalize input
        normalized = self.scaler.transform(past_points)
        
        # Prepare for model
        x = torch.FloatTensor(normalized).unsqueeze(0).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(x).cpu().numpy()[0]
        
        # Denormalize
        prediction = self.scaler.inverse_transform(prediction.reshape(1, -1))[0]
        
        return prediction
    
    def save(self, filepath):
        """Save model and scaler"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'sequence_length': self.sequence_length,
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model and scaler"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']
        self.sequence_length = checkpoint['sequence_length']
        self.is_fitted = True
        print(f"Model loaded from {filepath}")


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate haversine distance between two points in meters"""
    R = 6371000  # Earth radius in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c


if __name__ == "__main__":
    print("Trajectory Predictor Module")
    print("Import this module to use: from trajectory_predictor import TrajectoryPredictor")
