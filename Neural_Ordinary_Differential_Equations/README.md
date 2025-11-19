# Neural Ordinary Differential Equations

## Overview

Implementation of Neural Ordinary Differential Equations (Neural ODEs), a class of deep neural network models that use ODE solvers as part of the network architecture. Instead of specifying discrete layers, Neural ODEs parameterize the continuous dynamics of hidden states using an ODE specified by a neural network.

## Paper Reference

**Neural Ordinary Differential Equations**  
Authors: Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud  
Conference: NeurIPS 2018  
Paper: [arXiv:1806.07366](https://arxiv.org/abs/1806.07366)

**Key Contributions:**
- Continuous-depth neural networks using ODE solvers
- Constant memory cost for backpropagation
- Adaptive computation based on problem complexity
- Applications to time series, continuous normalizing flows, and more

## Project Structure

```
Neural_Ordinary_Differential_Equations/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── data/                        # Generated datasets
│   ├── mass_spring_oscillator_dataset.npz
│   └── dataset_info.txt
├── scripts/                     # Implementation scripts
│   ├── generate_dataset.py      # Dataset generation
│   └── train_neural_ode.py      # Model training
├── results/                     # Experimental results
│   ├── sample_trajectories.png
│   ├── neural_ode_predictions.png
│   ├── training_history.png
│   └── neural_ode_model.pt
└── docs/                        # Additional documentation
```

## Dataset

### Mass-Spring Oscillator System

The dataset consists of 1,000 periodic trajectories generated from a mass-spring oscillator with:

- **Variable Frequency:** Sampled uniformly from [0.5, 2.0]
- **Fixed Amplitude:** 1.0
- **Initial Conditions:** Sampled from standard Gaussian N(0,1)
- **Time Points:** 100 irregularly-sampled points per trajectory over [0, 10]
- **Noise:** Gaussian noise (σ=0.1) added to observations

**Dynamics:**
```
dx/dt = v
dv/dt = -ω²x - γv
```

where ω is the angular frequency and γ is the damping coefficient (set to 0 for undamped oscillation).

## Model Architecture

### Neural ODE Network

The model consists of:

1. **ODE Function:** 4-layer neural network (128 hidden units) with ELU activations
   - Input: 2D state (position, velocity)
   - Output: Time derivative of state
   
2. **ODE Solver:** Adaptive Dormand-Prince (dopri5) method
   - Relative tolerance: 1e-3
   - Absolute tolerance: 1e-4
   
3. **Direct State-Space Modeling:** No encoder/decoder bottleneck

**Model Parameters:** ~33,666 trainable parameters

### Key Features

- **Xavier Initialization:** Better gradient flow
- **ELU Activations:** Smooth gradients for ODE solving
- **AdamW Optimizer:** Weight decay for regularization
- **Learning Rate Scheduling:** Reduces LR on plateau

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- PyTorch 2.0+
- torchdiffeq 0.2.3+
- NumPy, SciPy, Matplotlib

## Usage

### 1. Generate Dataset

```bash
python scripts/generate_dataset.py
```

This creates:
- `data/mass_spring_oscillator_dataset.npz` - Complete dataset
- `data/dataset_info.txt` - Metadata
- `results/sample_trajectories.png` - Sample visualizations

### 2. Train Neural ODE Model

```bash
python scripts/train_neural_ode.py
```

**Training Configuration:**
- Epochs: 200
- Batch Size: 64
- Observation Points: 50 (randomly subsampled during training)
- Learning Rate: 1e-3
- Train/Test Split: 80/20

**Outputs:**
- `results/neural_ode_model.pt` - Saved model weights
- `results/training_history.png` - Loss and timing curves
- `results/neural_ode_predictions.png` - Predictions vs ground truth

### 3. Evaluate Model

The training script automatically evaluates on the test set and generates visualizations showing:
- Predicted trajectories vs true trajectories
- Position and velocity components
- Multiple test samples for comparison

## Results

### Performance

- **Test Loss (MSE):** ~0.01-0.02 (varies by run)
- **Training Time:** ~5-10 minutes on CPU for 200 epochs
- **Generalization:** Successfully reconstructs unseen trajectories with variable frequencies

### Key Observations

1. **Continuous Dynamics Learning:** Model learns smooth ODE dynamics from discrete observations
2. **Subsampling Robustness:** Trains on 50 random points, reconstructs full 100-point trajectories
3. **Frequency Generalization:** Handles variable frequencies despite training on mixed data
4. **Noise Robustness:** Maintains accuracy despite Gaussian observation noise

### Visualizations

- **Sample Trajectories:** Shows clean vs noisy observations
- **Predictions:** Compares model output with ground truth
- **Training Curves:** Demonstrates convergence and learning dynamics

## Implementation Details

### ODE Integration

Uses `torchdiffeq` library with adaptive step-size control:
- Automatically adjusts time steps based on solution dynamics
- Efficient for complex, stiff systems
- Supports backpropagation through ODE solver

### Training Strategy

- **Random Subsampling:** Different observation points each epoch
- **Full Reconstruction:** Model must interpolate/extrapolate to all time points
- **Adaptive Learning Rate:** Reduces when loss plateaus
- **Weight Decay:** Prevents overfitting

## Future Enhancements

- [ ] Implement continuous normalizing flows
- [ ] Add latent ODE for irregular time series
- [ ] Experiment with other ODE solvers (RK4, Euler, etc.)
- [ ] Add uncertainty quantification
- [ ] Test on real-world physics datasets

## References

```bibtex
@inproceedings{chen2018neural,
  title={Neural Ordinary Differential Equations},
  author={Chen, Ricky T. Q. and Rubanova, Yulia and Bettencourt, Jesse and Duvenaud, David},
  booktitle={Advances in Neural Information Processing Systems},
  year={2018}
}
```

## License

This implementation is open source and available under the MIT License.
