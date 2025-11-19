"""
Neural ODE Implementation

This script implements Neural Ordinary Differential Equations for two problems:
1. Time series forecasting on mass-spring oscillator data
2. Visualization of learned dynamics
Based on the paper: "Neural Ordinary Differential Equations" (Chen et al., NeurIPS 2018)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
import matplotlib.pyplot as plt
from pathlib import Path
import time
from tqdm import tqdm


class ODEFunc(nn.Module):
    """
    Neural network that defines the dynamics for the ODE solver.
    Learns the time derivative of the hidden state.
    """
    def __init__(self, hidden_dim=128):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2)
        )
        
        # Initialize with Xavier initialization
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, val=0)
    
    def forward(self, t, y):
        """
        Forward pass through the ODE function.
        
        Args:
            t: Current time (scalar)
            y: Current state (batch_size, 2)
        
        Returns:
            Time derivative dy/dt
        """
        return self.net(y)


class NeuralODE(nn.Module):
    """
    Neural ODE model that integrates an ODE defined by a neural network.
    """
    def __init__(self, func):
        super(NeuralODE, self).__init__()
        self.func = func
    
    def forward(self, y0, t):
        """
        Solve the ODE from initial condition y0 over times t.
        
        Args:
            y0: Initial state (batch_size, 2)
            t: Time points to evaluate (n_points,)
        
        Returns:
            Solution at time points t (n_points, batch_size, 2)
        """
        solution = odeint(self.func, y0, t, method='dopri5', rtol=1e-3, atol=1e-4)
        return solution


class ODENet(nn.Module):
    """
    Complete Neural ODE network - simplified to work directly in state space.
    """
    def __init__(self, input_dim=2, hidden_dim=128):
        super(ODENet, self).__init__()
        
        # ODE dynamics - work directly in state space
        self.ode_func = ODEFunc(hidden_dim)
        self.neural_ode = NeuralODE(self.ode_func)
    
    def forward(self, x0, t):
        """
        Forward pass through the full model.
        
        Args:
            x0: Initial observation (batch_size, input_dim)
            t: Time points (n_points,)
        
        Returns:
            Reconstructed trajectory (n_points, batch_size, input_dim)
        """
        # Solve ODE directly in state space
        x_t = self.neural_ode(x0, t)
        
        return x_t


def load_data(data_path='../data/mass_spring_oscillator_dataset.npz'):
    """Load the mass-spring oscillator dataset."""
    script_dir = Path(__file__).parent
    full_path = (script_dir / data_path).resolve()
    
    data = np.load(full_path)
    dataset = {
        'trajectories': torch.FloatTensor(data['trajectories']),
        'time_points': torch.FloatTensor(data['time_points']),
        'frequencies': torch.FloatTensor(data['frequencies']),
        'initial_states': torch.FloatTensor(data['initial_states'])
    }
    print(f"Loaded dataset: {dataset['trajectories'].shape[0]} trajectories")
    return dataset


def create_train_test_split(dataset, train_ratio=0.8):
    """Split dataset into training and testing sets."""
    n_total = dataset['trajectories'].shape[0]
    n_train = int(n_total * train_ratio)
    
    indices = torch.randperm(n_total)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    train_data = {
        'trajectories': dataset['trajectories'][train_indices],
        'time_points': dataset['time_points'][train_indices],
        'initial_states': dataset['initial_states'][train_indices]
    }
    
    test_data = {
        'trajectories': dataset['trajectories'][test_indices],
        'time_points': dataset['time_points'][test_indices],
        'initial_states': dataset['initial_states'][test_indices]
    }
    
    return train_data, test_data


def subsample_points(trajectories, time_points, n_obs=20):
    """
    Randomly subsample observation points from trajectories.
    
    Args:
        trajectories: Full trajectories (batch_size, n_points, 2)
        time_points: Time points (batch_size, n_points)
        n_obs: Number of observation points to subsample
    
    Returns:
        Subsampled data and indices
    """
    batch_size, n_points, _ = trajectories.shape
    
    # For each trajectory, randomly select n_obs points
    obs_trajectories = []
    obs_times = []
    obs_indices_list = []
    
    for i in range(batch_size):
        indices = torch.sort(torch.randperm(n_points)[:n_obs])[0]
        obs_trajectories.append(trajectories[i, indices])
        obs_times.append(time_points[i, indices])
        obs_indices_list.append(indices)
    
    return torch.stack(obs_trajectories), torch.stack(obs_times), obs_indices_list


def train_model(model, train_data, n_epochs=200, batch_size=64, n_obs=50, lr=1e-3, device='cpu'):
    """
    Train the Neural ODE model.
    
    Args:
        model: ODENet model
        train_data: Training dataset
        n_epochs: Number of training epochs
        batch_size: Batch size
        n_obs: Number of observation points per trajectory
        lr: Learning rate
        device: Device to train on
    
    Returns:
        Training history
    """
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.MSELoss()
    
    n_train = train_data['trajectories'].shape[0]
    n_batches = n_train // batch_size
    
    history = {'train_loss': [], 'epoch_time': []}
    
    print(f"\nTraining Neural ODE on {n_train} trajectories")
    print(f"Batch size: {batch_size}, Epochs: {n_epochs}, Learning rate: {lr}")
    print("="*70)
    
    for epoch in range(n_epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        
        # Shuffle training data
        perm = torch.randperm(n_train)
        
        for batch_idx in range(n_batches):
            # Get batch
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch_indices = perm[start_idx:end_idx]
            
            batch_traj = train_data['trajectories'][batch_indices].to(device)
            batch_times = train_data['time_points'][batch_indices].to(device)
            
            # Subsample observation points
            obs_traj, obs_times, _ = subsample_points(batch_traj, batch_times, n_obs)
            
            # Initial state
            x0 = obs_traj[:, 0, :].to(device)
            t = obs_times[0].to(device)  # Assume same time grid for batch
            
            # Forward pass
            optimizer.zero_grad()
            pred_traj = model(x0, t)  # (n_points, batch_size, 2)
            pred_traj = pred_traj.permute(1, 0, 2)  # (batch_size, n_points, 2)
            
            # Compute loss
            loss = criterion(pred_traj, obs_traj)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / n_batches
        
        history['train_loss'].append(avg_loss)
        history['epoch_time'].append(epoch_time)
        
        # Update learning rate
        scheduler.step(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{n_epochs}] - Loss: {avg_loss:.6f} - LR: {current_lr:.6f} - Time: {epoch_time:.2f}s")
    
    print("\nTraining completed!")
    return history


def evaluate_model(model, test_data, n_obs=20, device='cpu'):
    """
    Evaluate the Neural ODE model on test data.
    
    Args:
        model: Trained ODENet model
        test_data: Test dataset
        n_obs: Number of observation points
        device: Device to evaluate on
    
    Returns:
        Evaluation metrics
    """
    model.eval()
    model = model.to(device)
    criterion = nn.MSELoss()
    
    n_test = test_data['trajectories'].shape[0]
    total_loss = 0.0
    
    with torch.no_grad():
        for i in range(n_test):
            traj = test_data['trajectories'][i:i+1].to(device)
            times = test_data['time_points'][i:i+1].to(device)
            
            # Use all 100 points for evaluation
            x0 = traj[:, 0, :]
            t = times[0]
            
            # Forward pass
            pred_traj = model(x0, t)
            pred_traj = pred_traj.permute(1, 0, 2)
            
            # Compute loss
            loss = criterion(pred_traj, traj)
            total_loss += loss.item()
    
    avg_loss = total_loss / n_test
    print(f"\nTest Loss (MSE): {avg_loss:.6f}")
    
    return {'test_loss': avg_loss}


def visualize_predictions(model, test_data, n_samples=5, save_dir='../results'):
    """
    Visualize model predictions on test trajectories.
    
    Args:
        model: Trained model
        test_data: Test dataset
        n_samples: Number of samples to visualize
        save_dir: Directory to save figures
    """
    model.eval()
    script_dir = Path(__file__).parent
    save_path = (script_dir / save_dir).resolve()
    save_path.mkdir(parents=True, exist_ok=True)
    
    device = next(model.parameters()).device
    
    fig, axes = plt.subplots(n_samples, 2, figsize=(14, 3*n_samples))
    
    with torch.no_grad():
        for i in range(n_samples):
            idx = np.random.randint(0, test_data['trajectories'].shape[0])
            
            traj = test_data['trajectories'][idx:idx+1].to(device)
            times = test_data['time_points'][idx:idx+1].to(device)
            
            # Predict
            x0 = traj[:, 0, :]
            t = times[0]
            pred_traj = model(x0, t).permute(1, 0, 2).cpu().numpy()[0]
            
            true_traj = traj.cpu().numpy()[0]
            t_np = times.cpu().numpy()[0]
            
            # Plot position
            axes[i, 0].plot(t_np, true_traj[:, 0], 'b-', label='True', linewidth=2, alpha=0.7)
            axes[i, 0].plot(t_np, pred_traj[:, 0], 'r--', label='Predicted', linewidth=2, alpha=0.7)
            axes[i, 0].set_ylabel('Position', fontsize=11)
            axes[i, 0].set_title(f'Test Sample {idx} - Position', fontsize=12)
            axes[i, 0].legend()
            axes[i, 0].grid(True, alpha=0.3)
            
            # Plot velocity
            axes[i, 1].plot(t_np, true_traj[:, 1], 'b-', label='True', linewidth=2, alpha=0.7)
            axes[i, 1].plot(t_np, pred_traj[:, 1], 'r--', label='Predicted', linewidth=2, alpha=0.7)
            axes[i, 1].set_ylabel('Velocity', fontsize=11)
            axes[i, 1].set_title(f'Test Sample {idx} - Velocity', fontsize=12)
            axes[i, 1].legend()
            axes[i, 1].grid(True, alpha=0.3)
            
            if i == n_samples - 1:
                axes[i, 0].set_xlabel('Time', fontsize=11)
                axes[i, 1].set_xlabel('Time', fontsize=11)
    
    plt.tight_layout()
    output_path = save_path / 'neural_ode_predictions.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Predictions saved to {output_path}")
    plt.close()


def plot_training_history(history, save_dir='../results'):
    """Plot training loss over epochs."""
    script_dir = Path(__file__).parent
    save_path = (script_dir / save_dir).resolve()
    save_path.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curve
    axes[0].plot(history['train_loss'], 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Training Loss (MSE)', fontsize=12)
    axes[0].set_title('Training Loss Curve', fontsize=13)
    axes[0].grid(True, alpha=0.3)
    
    # Epoch time
    axes[1].plot(history['epoch_time'], 'g-', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Time (seconds)', fontsize=12)
    axes[1].set_title('Training Time per Epoch', fontsize=13)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = save_path / 'training_history.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Training history saved to {output_path}")
    plt.close()


def save_model(model, save_dir='../results'):
    """Save trained model."""
    script_dir = Path(__file__).parent
    save_path = (script_dir / save_dir).resolve()
    save_path.mkdir(parents=True, exist_ok=True)
    
    model_path = save_path / 'neural_ode_model.pt'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


def main():
    """Main training and evaluation pipeline."""
    print("\n" + "="*70)
    print("Neural ODE Implementation for Time Series Forecasting")
    print("="*70)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load data
    print("\n1. Loading dataset...")
    dataset = load_data()
    
    # Split data
    print("\n2. Creating train/test split...")
    train_data, test_data = create_train_test_split(dataset, train_ratio=0.8)
    print(f"Training samples: {train_data['trajectories'].shape[0]}")
    print(f"Test samples: {test_data['trajectories'].shape[0]}")
    
    # Initialize model
    print("\n3. Initializing Neural ODE model...")
    model = ODENet(input_dim=2, hidden_dim=128)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train model
    print("\n4. Training model...")
    history = train_model(
        model, 
        train_data, 
        n_epochs=200, 
        batch_size=64, 
        n_obs=50,  # Use 50 random points for training
        lr=1e-3,
        device=device
    )
    
    # Evaluate model
    print("\n5. Evaluating model on test set...")
    metrics = evaluate_model(model, test_data, device=device)
    
    # Visualize results
    print("\n6. Generating visualizations...")
    plot_training_history(history)
    visualize_predictions(model, test_data, n_samples=5)
    
    # Save model
    print("\n7. Saving model...")
    save_model(model)
    
    print("\n" + "="*70)
    print("Experiment completed successfully!")
    print("="*70)
    
    return model, history, metrics


if __name__ == "__main__":
    model, history, metrics = main()
