# Linear Algebra and Matrix Methods

## Overview

Comprehensive implementation of fundamental linear algebra operations, matrix decomposition methods, and matrix calculus using PyTorch for deep learning and scientific computing applications.

## Reference Materials

This project is based on the following reference materials (available in `docs/` folder):

1. **Linear Algebra Review**  
   [Linear Algebra Review and Reference](https://www.cs.cmu.edu/~zkolter/course/15-884/linalg-review.pdf) - Zico Kolter, CMU  
   Comprehensive review of linear algebra fundamentals for machine learning

2. **Matrix Calculus**  
   [The Matrix Calculus You Need For Deep Learning](https://arxiv.org/abs/1802.01528) - arXiv:1802.01528  
   Terence Parr and Jeremy Howard  
   Complete guide to matrix calculus for deep learning applications

## Jupyter Notebooks

This project includes three interactive Jupyter notebooks for hands-on learning:

### 1. Tensor Primer (`tensor_primer.ipynb`)

Complete PyTorch tensor tutorial with individual executable cells covering:
- **Tensor Basics:** Creation, types, properties, and operations
- **Reshaping & Indexing:** Views, reshape, squeeze, unsqueeze, slicing
- **Mathematical Operations:** Element-wise, reduction, and broadcasting
- **Linear Algebra:** Matrix multiplication, transpose, norms
- **Advanced Topics:** Eigenvalues, SVD, QR, Cholesky decomposition
- **Automatic Differentiation:** Gradients and computational graphs
- **Device Management:** CPU/GPU operations and memory optimization

### 2. Matrix Decompositions (`matrix_decompositions.ipynb`)

Implementations of major matrix decomposition methods:
- **LU Decomposition:** Lower-upper factorization
- **QR Decomposition:** Orthogonal-triangular factorization
- **SVD:** Singular Value Decomposition and applications
- **Eigenvalue Decomposition:** Spectral analysis
- **Cholesky Decomposition:** Positive definite matrices
- **Low-Rank Approximations:** Data compression techniques
- **Image Compression:** Practical SVD application
- **Linear Systems:** Solving Ax = b using decompositions
- **Matrix Properties:** Rank, condition number, pseudo-inverse

### 3. Matrix Calculus (`matrix_calculus.ipynb`)

Comprehensive guide to matrix calculus and optimization:
- **Gradients:** Scalar, vector, and matrix derivatives
- **Jacobian Matrix:** Vector-valued function derivatives
- **Hessian Matrix:** Second-order derivatives
- **Chain Rule:** Composite and nested functions
- **Directional Derivatives:** Gradients along specific directions
- **Optimization Methods:** Gradient descent and Newton's method
- **Backpropagation:** Neural network automatic differentiation
- **Common Formulas:** Standard matrix derivative identities
- **Visualizations:** Gradient fields, contours, and 3D surfaces
- **Applications:** Linear regression with full training loop

## Contents

- `docs/` - Documentation and mathematical background (Linear_Algebra.pdf)
- `data/` - Generated datasets and examples
- `scripts/` - Implementation scripts and Jupyter notebooks
- `results/` - Experimental results and visualizations

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Activate your Python environment:
   ```bash
   source venv/bin/activate  # Linux/Mac
   .\venv\Scripts\Activate.ps1  # Windows PowerShell
   ```

2. Launch Jupyter Notebook or open in VS Code:
   ```bash
   jupyter notebook
   ```

3. Open any of the three notebooks in the `scripts/` folder and run cells interactively.

## Applications

- Principal Component Analysis (PCA)
- Singular Value Decomposition for data compression
- Image compression and low-rank approximations
- Matrix factorization methods
- Numerical linear algebra techniques
- Optimization algorithms for machine learning
- Gradient-based learning and backpropagation

## License

This project is open source and available under the MIT License.
