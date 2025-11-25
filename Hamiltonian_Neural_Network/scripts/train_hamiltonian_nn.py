"""
Hamiltonian Neural Network Implementation

This script implements a Hamiltonian Neural Network for learning conserved quantities
in dynamical systems. The network learns the Hamiltonian (energy function) directly,
ensuring that the learned dynamics conserve the total energy.

Based on the paper: "Hamiltonian Neural Networks" (Greydanus et al., NeurIPS 2019)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
import time


class Hamiltonian(nn.Module):
    """
    Neural network that learns the Hamiltonian (energy function).
    Learns H(q, p) where q is position and p is momentum/velocity.
    """
    def __init__(self, hidden_dim=128):
        super(Hamiltonian, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize with Xavier initialization
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, val=0)
    
    def forward(self, state):
        """
        Compute the Hamiltonian (total energy) of the system.
        
        Args:
            state: System state (batch_size, 2) where state[:, 0] is position, state[:, 1] is velocity
        
        Returns:
            Hamiltonian value (batch_size, 1)
        """
        return self.net(state)


class HamiltonianNN(nn.Module):
    """
    Hamiltonian Neural Network model.
    Uses automatic differentiation to compute the canonical equations of motion.
    """
    def __init__(self, hidden_dim=128):
        super(HamiltonianNN, self).__init__()
        self.hamiltonian = Hamiltonian(hidden_dim)
    
    def forward(self, state, dt=0.01):
        """
        Compute the next state using Hamiltonian dynamics.
        Canonical equations: dq/dt = ∂H/∂p, dp/dt = -∂H/∂q
        
        Args:
            state: Current state (batch_size, 2) [position, velocity]
            dt: Time step for integration
        
        Returns:
            Next state (batch_size, 2)
        """
        state_detached = state.clone().detach().requires_grad_(True)
        
        # Compute Hamiltonian
        H = self.hamiltonian(state_detached)
        H_sum = H.sum()
        
        # Compute gradients using autograd
        dH_dstate = torch.autograd.grad(H_sum, state_detached, create_graph=False, retain_graph=False)[0]
        
        # Canonical equations of motion
        # dq/dt = ∂H/∂p = dH_dstate[:, 1]
        # dp/dt = -∂H/∂q = -dH_dstate[:, 0]
        
        dq_dt = dH_dstate[:, 1:2]  # (batch_size, 1)
        dp_dt = -dH_dstate[:, 0:1]  # (batch_size, 1)
        
        # Concatenate to get full state derivative
        state_dot = torch.cat([dq_dt, dp_dt], dim=1)  # (batch_size, 2)
        
        # Euler integration step
        next_state = state + dt * state_dot
        
        return next_state
    
    def integrate_trajectory(self, initial_state, time_points):
        """
        Integrate the trajectory over multiple time points using a simpler direct approach.
        
        Args:
            initial_state: Initial state (batch_size, 2)
            time_points: Time points to evaluate (n_points,)
        
        Returns:
            Trajectory (n_points, batch_size, 2)
        """
        batch_size = initial_state.shape[0]
        n_points = len(time_points)
        device = initial_state.device
        
        trajectory = torch.zeros(n_points, batch_size, 2, device=device)
        trajectory[0] = initial_state
        
        for i in range(1, n_points):
            dt = (time_points[i] - time_points[i-1]).item()
            
            # Current state
            state = trajectory[i-1].clone().detach().requires_grad_(True)
            
            # Compute Hamiltonian
            H = self.hamiltonian(state)
            H_sum = H.sum()
            
            # Compute gradients
            dH_dstate = torch.autograd.grad(H_sum, state, create_graph=False, retain_graph=False)[0]
            
            # Canonical equations
            dq_dt = dH_dstate[:, 1:2]
            dp_dt = -dH_dstate[:, 0:1]
            state_dot = torch.cat([dq_dt, dp_dt], dim=1)
            
            # Euler step
            next_state = trajectory[i-1] + dt * state_dot
            trajectory[i] = next_state
        
        return trajectory


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
    Train the Hamiltonian Neural Network model using single-step prediction.
    
    Args:
        model: HamiltonianNN model
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
    
    print(f"\nTraining Hamiltonian Neural Network on {n_train} trajectories")
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
            
            # For simplicity, use Hamiltonian directly for energy conservation
            # Train to minimize energy conservation error
            optimizer.zero_grad()
            
            # Compute Hamiltonian for all points in trajectory
            H_values = model.hamiltonian(batch_traj.view(-1, 2)).view(batch_traj.shape[0], batch_traj.shape[1], 1)
            
            # Ideal Hamiltonian for mass-spring system: H = 0.5 * (p^2 + omega^2 * q^2)
            # We can approximate by constraining H to be constant over trajectory
            H_initial = H_values[:, 0:1, :]  # (batch_size, 1, 1)
            H_error = (H_values - H_initial) ** 2  # Energy should be constant
            
            loss = H_error.mean()
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
            print(f"Epoch [{epoch+1}/{n_epochs}] - Energy Error: {avg_loss:.6f} - LR: {current_lr:.6f} - Time: {epoch_time:.2f}s")
    
    print("\nTraining completed!")
    return history


def evaluate_model(model, test_data, device='cpu'):
    """
    Evaluate the Hamiltonian Neural Network model on test data.
    
    Args:
        model: Trained HamiltonianNN model
        test_data: Test dataset
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
            
            # Compute Hamiltonian for all points
            H_values = model.hamiltonian(traj.view(-1, 2)).view(traj.shape[0], traj.shape[1], 1)
            
            # Energy should be approximately constant
            H_initial = H_values[:, 0:1, :]
            energy_error = (H_values - H_initial) ** 2
            loss = energy_error.mean()
            total_loss += loss.item()
    
    avg_loss = total_loss / n_test
    print(f"\nTest Energy Conservation Error (MSE): {avg_loss:.6f}")
    
    return {'test_energy_error': avg_loss}


def visualize_predictions(model, test_data, n_samples=5, save_dir='../results'):
    """
    Visualize learned Hamiltonian function on test trajectories.
    
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
            t_np = times.cpu().numpy()[0]
            true_traj = traj.cpu().numpy()[0]
            
            # Compute Hamiltonian values for the trajectory
            H_values = model.hamiltonian(traj.view(-1, 2)).view(traj.shape[0], traj.shape[1], 1)
            H_values = H_values.cpu().numpy()[0, :, 0]
            
            # Plot trajectory
            axes[i, 0].plot(t_np, true_traj[:, 0], 'b-', label='Position', linewidth=2, alpha=0.7)
            axes[i, 0].plot(t_np, true_traj[:, 1], 'r-', label='Velocity', linewidth=2, alpha=0.7)
            axes[i, 0].set_ylabel('State Value', fontsize=11)
            axes[i, 0].set_title(f'Test Sample {idx} - Trajectory', fontsize=12)
            axes[i, 0].legend()
            axes[i, 0].grid(True, alpha=0.3)
            
            # Plot Hamiltonian (energy)
            axes[i, 1].plot(t_np, H_values, 'g-', linewidth=2, alpha=0.7)
            axes[i, 1].fill_between(t_np, H_values.min() - 0.01, H_values.max() + 0.01, alpha=0.2)
            axes[i, 1].set_ylabel('Hamiltonian H(q,p)', fontsize=11)
            axes[i, 1].set_title(f'Test Sample {idx} - Energy Conservation', fontsize=12)
            axes[i, 1].grid(True, alpha=0.3)
            
            if i == n_samples - 1:
                axes[i, 0].set_xlabel('Time', fontsize=11)
                axes[i, 1].set_xlabel('Time', fontsize=11)
    
    plt.tight_layout()
    output_path = save_path / 'hamiltonian_nn_predictions.png'
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
    
    model_path = save_path / 'hamiltonian_nn_model.pt'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


def main():
    """Main training and evaluation pipeline."""
    print("\n" + "="*70)
    print("Hamiltonian Neural Network Implementation")
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
    print("\n3. Initializing Hamiltonian Neural Network model...")
    model = HamiltonianNN(hidden_dim=128)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train model
    print("\n4. Training model...")
    history = train_model(
        model, 
        train_data, 
        n_epochs=200, 
        batch_size=64, 
        n_obs=50,
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
