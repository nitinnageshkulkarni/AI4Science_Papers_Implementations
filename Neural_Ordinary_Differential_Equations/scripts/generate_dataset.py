"""
Mass-Spring Oscillator Dataset Generator

This script generates synthetic data for training Neural ODEs on periodic trajectories.
Creates 1,000 periodic trajectories with variable frequency and the same amplitude.

Dataset specifications:
- 1,000 trajectories
- Variable frequency, same amplitude
- Initial points sampled from standard Gaussian
- Gaussian noise added to observations
- 100 irregularly-sampled time points per trajectory
- Training: subsample random points, reconstruct full 100 points
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path


class MassSpringOscillator:
    """Mass-spring oscillator dynamics for generating periodic trajectories."""
    
    def __init__(self, frequency, amplitude=1.0, damping=0.0):
        """
        Initialize oscillator parameters.
        
        Args:
            frequency (float): Angular frequency (omega)
            amplitude (float): Amplitude of oscillation
            damping (float): Damping coefficient
        """
        self.omega = frequency
        self.amplitude = amplitude
        self.damping = damping
    
    def dynamics(self, t, state):
        """
        Define the ODE dynamics: d/dt [x, v] = [v, -omega^2 * x - damping * v]
        
        Args:
            t (float): Time
            state (array): [position, velocity]
        
        Returns:
            array: Time derivatives [dx/dt, dv/dt]
        """
        x, v = state
        dxdt = v
        dvdt = -self.omega**2 * x - self.damping * v
        return np.array([dxdt, dvdt])
    
    def solve(self, t_points, initial_state):
        """
        Solve the ODE using simple Euler integration.
        
        Args:
            t_points (array): Time points to evaluate
            initial_state (array): Initial [position, velocity]
        
        Returns:
            array: Trajectory of shape (n_points, 2)
        """
        trajectory = np.zeros((len(t_points), 2))
        trajectory[0] = initial_state
        
        for i in range(1, len(t_points)):
            dt = t_points[i] - t_points[i-1]
            state = trajectory[i-1]
            # RK4 integration for better accuracy
            k1 = self.dynamics(t_points[i-1], state)
            k2 = self.dynamics(t_points[i-1] + dt/2, state + dt*k1/2)
            k3 = self.dynamics(t_points[i-1] + dt/2, state + dt*k2/2)
            k4 = self.dynamics(t_points[i], state + dt*k3)
            trajectory[i] = state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
        
        return trajectory


def generate_irregular_time_points(n_points=100, t_max=10.0):
    """
    Generate irregularly-sampled time points.
    
    Args:
        n_points (int): Number of time points
        t_max (float): Maximum time value
    
    Returns:
        array: Sorted irregular time points
    """
    # Generate random time points and sort them
    t_points = np.sort(np.random.uniform(0, t_max, n_points))
    t_points[0] = 0  # Ensure we start at t=0
    return t_points


def generate_dataset(n_trajectories=1000, 
                    n_points=100, 
                    amplitude=1.0,
                    freq_range=(0.5, 2.0),
                    noise_std=0.1,
                    t_max=10.0,
                    seed=42):
    """
    Generate mass-spring oscillator dataset.
    
    Args:
        n_trajectories (int): Number of trajectories to generate
        n_points (int): Number of time points per trajectory
        amplitude (float): Amplitude of oscillations (fixed)
        freq_range (tuple): Range of frequencies (min, max)
        noise_std (float): Standard deviation of observation noise
        t_max (float): Maximum time value
        seed (int): Random seed for reproducibility
    
    Returns:
        dict: Dataset containing trajectories, time points, and metadata
    """
    np.random.seed(seed)
    
    dataset = {
        'trajectories': [],
        'time_points': [],
        'frequencies': [],
        'initial_states': [],
        'clean_trajectories': []  # Store noise-free trajectories for comparison
    }
    
    for i in range(n_trajectories):
        # Sample frequency uniformly from range
        frequency = np.random.uniform(freq_range[0], freq_range[1])
        
        # Sample initial state from standard Gaussian
        initial_position = np.random.randn()
        initial_velocity = np.random.randn()
        initial_state = np.array([initial_position, initial_velocity])
        
        # Generate irregular time points
        t_points = generate_irregular_time_points(n_points, t_max)
        
        # Create oscillator and solve
        oscillator = MassSpringOscillator(frequency=frequency, amplitude=amplitude)
        clean_trajectory = oscillator.solve(t_points, initial_state)
        
        # Add Gaussian noise to observations
        noisy_trajectory = clean_trajectory + np.random.randn(*clean_trajectory.shape) * noise_std
        
        # Store data
        dataset['trajectories'].append(noisy_trajectory)
        dataset['time_points'].append(t_points)
        dataset['frequencies'].append(frequency)
        dataset['initial_states'].append(initial_state)
        dataset['clean_trajectories'].append(clean_trajectory)
        
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{n_trajectories} trajectories")
    
    # Convert lists to numpy arrays
    dataset['trajectories'] = np.array(dataset['trajectories'])
    dataset['time_points'] = np.array(dataset['time_points'])
    dataset['frequencies'] = np.array(dataset['frequencies'])
    dataset['initial_states'] = np.array(dataset['initial_states'])
    dataset['clean_trajectories'] = np.array(dataset['clean_trajectories'])
    
    return dataset


def visualize_samples(dataset, n_samples=5, save_path='../results/sample_trajectories.png'):
    """
    Visualize sample trajectories from the dataset.
    
    Args:
        dataset (dict): Dataset dictionary
        n_samples (int): Number of samples to visualize
        save_path (str): Path to save the figure (relative to script location)
    """
    # Get absolute path relative to the script location
    if save_path:
        script_dir = Path(__file__).parent
        save_path = (script_dir / save_path).resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(n_samples, 2, figsize=(12, 3*n_samples))
    
    for i in range(n_samples):
        idx = np.random.randint(0, len(dataset['trajectories']))
        t = dataset['time_points'][idx]
        traj_noisy = dataset['trajectories'][idx]
        traj_clean = dataset['clean_trajectories'][idx]
        freq = dataset['frequencies'][idx]
        
        # Plot position
        axes[i, 0].plot(t, traj_clean[:, 0], 'b-', label='Clean', alpha=0.7)
        axes[i, 0].scatter(t, traj_noisy[:, 0], c='r', s=10, label='Noisy', alpha=0.6)
        axes[i, 0].set_ylabel('Position')
        axes[i, 0].set_title(f'Trajectory {idx} (freq={freq:.2f})')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        
        # Plot velocity
        axes[i, 1].plot(t, traj_clean[:, 1], 'b-', label='Clean', alpha=0.7)
        axes[i, 1].scatter(t, traj_noisy[:, 1], c='r', s=10, label='Noisy', alpha=0.6)
        axes[i, 1].set_ylabel('Velocity')
        axes[i, 1].set_title(f'Trajectory {idx} (freq={freq:.2f})')
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)
        
        if i == n_samples - 1:
            axes[i, 0].set_xlabel('Time')
            axes[i, 1].set_xlabel('Time')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    plt.close()


def save_dataset(dataset, save_dir='../data'):
    """
    Save dataset to disk.
    
    Args:
        dataset (dict): Dataset dictionary
        save_dir (str): Directory to save the dataset
    """
    # Get absolute path relative to the script location
    script_dir = Path(__file__).parent
    save_path = (script_dir / save_dir).resolve()
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save as compressed numpy file
    np.savez_compressed(
        save_path / 'mass_spring_oscillator_dataset.npz',
        trajectories=dataset['trajectories'],
        time_points=dataset['time_points'],
        frequencies=dataset['frequencies'],
        initial_states=dataset['initial_states'],
        clean_trajectories=dataset['clean_trajectories']
    )
    
    print(f"Dataset saved to {save_path / 'mass_spring_oscillator_dataset.npz'}")
    
    # Save metadata
    metadata = {
        'n_trajectories': len(dataset['trajectories']),
        'n_points_per_trajectory': dataset['trajectories'].shape[1],
        'frequency_range': (dataset['frequencies'].min(), dataset['frequencies'].max()),
        'frequency_mean': dataset['frequencies'].mean(),
        'frequency_std': dataset['frequencies'].std()
    }
    
    with open(save_path / 'dataset_info.txt', 'w') as f:
        f.write("Mass-Spring Oscillator Dataset\n")
        f.write("="*50 + "\n\n")
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Metadata saved to {save_path / 'dataset_info.txt'}")


def load_dataset(data_path='../data/mass_spring_oscillator_dataset.npz'):
    """
    Load dataset from disk.
    
    Args:
        data_path (str): Path to the dataset file
    
    Returns:
        dict: Dataset dictionary
    """
    data = np.load(data_path)
    dataset = {
        'trajectories': data['trajectories'],
        'time_points': data['time_points'],
        'frequencies': data['frequencies'],
        'initial_states': data['initial_states'],
        'clean_trajectories': data['clean_trajectories']
    }
    print(f"Dataset loaded from {data_path}")
    print(f"Number of trajectories: {len(dataset['trajectories'])}")
    return dataset


def subsample_trajectory(trajectory, time_points, n_subsample):
    """
    Subsample a trajectory for training (randomly select subset of points).
    
    Args:
        trajectory (array): Full trajectory of shape (n_points, 2)
        time_points (array): Time points of shape (n_points,)
        n_subsample (int): Number of points to subsample
    
    Returns:
        tuple: (subsampled_trajectory, subsampled_times, subsample_indices)
    """
    n_points = len(time_points)
    subsample_indices = np.sort(np.random.choice(n_points, n_subsample, replace=False))
    
    return (trajectory[subsample_indices], 
            time_points[subsample_indices], 
            subsample_indices)


if __name__ == "__main__":
    print("Generating Mass-Spring Oscillator Dataset...")
    print("="*60)
    
    # Generate dataset
    dataset = generate_dataset(
        n_trajectories=1000,
        n_points=100,
        amplitude=1.0,
        freq_range=(0.5, 2.0),
        noise_std=0.1,
        t_max=10.0,
        seed=42
    )
    
    print("\nDataset generation complete!")
    print(f"Shape of trajectories: {dataset['trajectories'].shape}")
    print(f"Shape of time points: {dataset['time_points'].shape}")
    print(f"Frequency range: [{dataset['frequencies'].min():.2f}, {dataset['frequencies'].max():.2f}]")
    
    # Save dataset
    save_dataset(dataset)
    
    # Visualize samples
    print("\nGenerating visualizations...")
    visualize_samples(dataset, n_samples=5)
    
    # Test subsampling
    print("\nTesting subsampling functionality...")
    sample_traj = dataset['trajectories'][0]
    sample_times = dataset['time_points'][0]
    sub_traj, sub_times, indices = subsample_trajectory(sample_traj, sample_times, n_subsample=30)
    print(f"Original points: {len(sample_traj)}, Subsampled points: {len(sub_traj)}")
    
    print("\nDataset generation complete! Ready for Neural ODE training.")
