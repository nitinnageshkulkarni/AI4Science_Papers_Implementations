# Hamiltonian Neural Network

**Paper:** Hamiltonian Neural Networks (Greydanus et al., NeurIPS 2019)  
**Link:** [arXiv:1906.01563](https://arxiv.org/abs/1906.01563) | [PDF](docs/Hamiltonian%20Neural%20Network.pdf)

Implementation of Hamiltonian Neural Networks for learning conserved quantities in dynamical systems. Unlike standard neural networks that learn arbitrary dynamics, HNNs learn the Hamiltonian (energy function) directly, ensuring that the learned dynamics conserve total energy and obey symplectic geometry principles.

## Contents

- `data/` - Dataset files for training and evaluation
- `docs/` - Documentation and references
- `results/` - Model outputs and training results
- `scripts/` - Python scripts for implementation

## Key Features

- **Symplectic Geometry:** Learns dynamics that preserve the canonical structure of Hamiltonian systems
- **Energy Conservation:** Automatically respects conservation of energy in learned dynamics
- **Physics-Informed Learning:** Incorporates physical laws directly into the neural network architecture
- **Automatic Differentiation:** Uses PyTorch's autograd to compute canonical equations of motion

## Dataset

The implementation uses a mass-spring oscillator dataset with:
- **1,000 trajectories** with variable frequency and fixed amplitude
- **100 irregularly-sampled time points** per trajectory
- **Gaussian noise** added to observations
- **States:** [position, velocity]

## Model Architecture

### Hamiltonian Network
- Input: State vector [q, p] (position and momentum)
- 4-layer fully connected network with ELU activations
- Output: Scalar Hamiltonian H(q, p)
- Uses automatic differentiation to compute canonical equations

### Training
- **Optimizer:** AdamW with weight decay regularization
- **Learning rate:** 1e-3 with learning rate scheduling
- **Training epochs:** 200
- **Batch size:** 64
- **Loss:** Mean Squared Error (MSE)

## Usage

1. Generate dataset:
   ```bash
   python scripts/generate_dataset.py
   ```

2. Train the model:
   ```bash
   python scripts/train_hamiltonian_nn.py
   ```

## Results

The trained model achieves:
- **Test Loss:** Converges to ~1e-4 MSE on held-out trajectories
- **Energy Conservation:** Learned dynamics preserve energy significantly better than standard NNs
- **Long-term Stability:** Accurate predictions even for long time horizons

Output files:
- `results/hamiltonian_nn_model.pt` - Trained model weights
- `results/hamiltonian_nn_predictions.png` - Sample predictions vs ground truth
- `results/training_history.png` - Training curves

## References

- Greydanus, S., Dzamba, M., & Yildirim, A. (2019). Hamiltonian Neural Networks. NeurIPS 2019.
- Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. K. (2019). Neural Ordinary Differential Equations. NeurIPS 2019.

