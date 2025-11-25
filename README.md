c:\Users\nitin\Desktop\AI4Science_Papers_Implementations\Hamiltonian_Neural_Network\venv\Scripts\python.exe generate_dataset.py# AI4Science Papers Implementations

This repository contains implementations of various AI4Science papers and miscellaneous implementations that work on the intersection of AI and scientific computing.

## About

This project focuses on bridging artificial intelligence with scientific research by implementing papers chronologically as developments emerge in the field, exploring novel approaches at the intersection of AI and science.

## Implementations

### 1. [Linear Algebra and Matrix Methods](./Linear_Algebra_and_Matrix_Methods)

Comprehensive implementation of fundamental linear algebra operations and matrix methods using PyTorch. This project includes:

- **Tensor Primer:** Complete PyTorch tensor tutorial covering:
  - Tensor creation, operations, and broadcasting
  - Linear algebra operations and matrix computations
  - Automatic differentiation and gradient computation
  - Device management and optimization techniques

- **Matrix Decompositions:** Implementation of major decomposition methods:
  - LU, QR, SVD, Eigenvalue, and Cholesky decompositions
  - Low-rank approximations and applications
  - Image compression using SVD
  - Solving linear systems and computing matrix properties

- **Matrix Calculus:** Comprehensive guide to calculus with matrices:
  - Gradients, Jacobians, and Hessians
  - Automatic differentiation and backpropagation
  - Directional derivatives and optimization methods
  - Gradient descent and Newton's method
  - Practical applications in machine learning

### 2. [Neural Ordinary Differential Equations](./Neural_Ordinary_Differential_Equations)

**Paper:** Neural Ordinary Differential Equations (Chen et al., NeurIPS 2018)  
**Link:** [arXiv:1806.07366](https://arxiv.org/abs/1806.07366)

Implementation of Neural ODEs for modeling continuous-depth neural networks using ODE solvers. This project includes:

- **Dataset Generation:** Mass-spring oscillator system with 1,000 periodic trajectories
  - Variable frequency, fixed amplitude
  - 100 irregularly-sampled time points per trajectory
  - Gaussian noise added to observations
  
- **Model Architecture:** 
  - Direct state-space modeling without encoder/decoder bottleneck
  - Adaptive ODE solver (dopri5) with neural network dynamics
  - Deep neural network (4 layers, 128 hidden units)
  
- **Training & Evaluation:**
  - Time series forecasting and reconstruction
  - Subsampling during training (50 random points)
  - Full trajectory reconstruction (100 points)
  - Comprehensive visualization of predictions vs ground truth

### 3. [Hamiltonian Neural Networks](./Hamiltonian_Neural_Network)

**Paper:** Hamiltonian Neural Networks (Greydanus et al., NeurIPS 2019)  
**Link:** [arXiv:1906.01563](https://arxiv.org/abs/1906.01563)

Implementation of Hamiltonian Neural Networks that learn the energy function of dynamical systems, ensuring conservation of energy and symplectic structure. This project includes:

- **Physics-Informed Learning:**
  - Learns the Hamiltonian (energy function) directly
  - Automatic differentiation to compute canonical equations of motion
  - Preserves energy conservation in learned dynamics
  - Respects symplectic geometry principles

- **Dataset:** Mass-spring oscillator system (shared with Neural ODE)
  - 1,000 trajectories with variable frequency
  - Energy conservation properties validated
  
- **Model Architecture:**
  - Neural network that outputs scalar Hamiltonian H(q,p)
  - 4-layer fully connected network (128 hidden units)
  - Uses PyTorch autograd for gradient computation
  
- **Training Approach:**
  - Energy conservation loss: minimizes variation in Hamiltonian over trajectory
  - AdamW optimizer with learning rate scheduling
  - Converges to near-zero energy variation
  
- **Results:**
  - Successfully learns to conserve energy
  - Test energy conservation error: < 1e-6
  - Visualizations of learned energy function and trajectory dynamics

---

## Project Structure

Each implementation follows a consistent structure:
```
Project_Name/
├── README.md          # Project-specific documentation
├── requirements.txt   # Python dependencies
├── data/             # Datasets and data generation
├── scripts/          # Implementation scripts
├── results/          # Experimental results and visualizations
└── docs/             # Additional documentation
```

## Contributing

Contributions, issues, and feature requests are welcome!

## License

This project is open source and available under the MIT License.
