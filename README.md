This repository contains implementations of various AI4Science papers and miscellaneous implementations that work on the intersection of AI and scientific computing.

## About

This project focuses on bridging artificial intelligence with scientific research by implementing papers chronologically as developments emerge in the field, exploring novel approaches at the intersection of AI and science.

## Implementations

| # | Project | Paper | Key Concepts | Status |
|---|---------|-------|--------------|--------|
| 1 | **[Linear Algebra & Matrix Methods](./Linear_Algebra_and_Matrix_Methods)** | [Linear Algebra](./Linear_Algebra_and_Matrix_Methods/docs/Linear_Algebra.pdf) \| [Matrix Calculus](./Linear_Algebra_and_Matrix_Methods/docs/The%20Matrix%20Calculus%20You%20Need%20For%20Deep%20Learning.pdf) | Tensor operations, decompositions, calculus with matrices, autograd | ✅ Complete |
| 2 | **[Neural Ordinary Differential Equations](./Neural_Ordinary_Differential_Equations)** | [Chen et al., NeurIPS 2018](./Neural_Ordinary_Differential_Equations/docs/Neural%20Ordinary%20Differential%20Equations.pdf) | Continuous-depth networks, ODE solvers, time series modeling | ✅ Complete |
| 3 | **[Hamiltonian Neural Networks](./Hamiltonian_Neural_Network)** | [Greydanus et al., NeurIPS 2019](./Hamiltonian_Neural_Network/docs/Hamiltonian%20Neural%20Network.pdf) | Physics-informed learning, energy conservation, symplectic geometry | ✅ Complete |

### Detailed Descriptions

#### 1. Linear Algebra and Matrix Methods

**Papers:** [Linear Algebra](./Linear_Algebra_and_Matrix_Methods/docs/Linear_Algebra.pdf) | [Matrix Calculus](./Linear_Algebra_and_Matrix_Methods/docs/The%20Matrix%20Calculus%20You%20Need%20For%20Deep%20Learning.pdf)

Comprehensive implementation of fundamental linear algebra operations and matrix methods using PyTorch:

- **Tensor Primer:** Complete PyTorch tensor tutorial
- **Matrix Decompositions:** LU, QR, SVD, Eigenvalue, Cholesky decompositions
- **Matrix Calculus:** Gradients, Jacobians, Hessians, automatic differentiation

#### 2. Neural Ordinary Differential Equations

**Paper:** [Chen et al., NeurIPS 2018](./Neural_Ordinary_Differential_Equations/docs/Neural%20Ordinary%20Differential%20Equations.pdf) | [arXiv](https://arxiv.org/abs/1806.07366)

Implementation of Neural ODEs for modeling continuous-depth neural networks:

- **Dataset:** Mass-spring oscillator (1,000 trajectories, 100 time points)
- **Model:** Adaptive ODE solver with neural network dynamics (4 layers, 128 hidden units)
- **Training:** Time series forecasting with energy conservation validation

#### 3. Hamiltonian Neural Networks

**Paper:** [Greydanus et al., NeurIPS 2019](./Hamiltonian_Neural_Network/docs/Hamiltonian%20Neural%20Network.pdf) | [arXiv](https://arxiv.org/abs/1906.01563)

Implementation of Hamiltonian Neural Networks for physics-informed learning:

- **Approach:** Learns the Hamiltonian (energy function) directly to ensure energy conservation
- **Dataset:** Mass-spring oscillator (1,000 trajectories with variable frequency)
- **Results:** Energy conservation error < 1e-6 on test set

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
