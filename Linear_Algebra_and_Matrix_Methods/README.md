# Linear Algebra and Matrix Methods

## Overview

Comprehensive implementation of fundamental linear algebra operations, matrix decomposition methods, and matrix calculus using PyTorch for deep learning and scientific computing applications.

This project serves as a foundational resource for understanding the mathematical principles underlying modern machine learning and scientific computing, with hands-on implementations and visualizations.

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

### Machine Learning
- Principal Component Analysis (PCA) for dimensionality reduction
- Singular Value Decomposition for recommender systems
- Matrix factorization for collaborative filtering
- Feature extraction and data preprocessing

### Computer Vision
- Image compression using low-rank approximations
- Image denoising and reconstruction
- Facial recognition (eigenfaces)

### Optimization
- Gradient descent and variants
- Newton's method and quasi-Newton methods
- Constrained optimization with Lagrange multipliers
- Backpropagation in neural networks

### Scientific Computing
- Solving linear systems Ax = b
- Least squares regression
- Eigenvalue problems
- Numerical stability analysis

## Key Learning Outcomes

After working through these notebooks, you will understand:

1. **Tensor Operations:** How to efficiently manipulate multi-dimensional arrays
2. **Matrix Decompositions:** When and how to apply different factorization methods
3. **Matrix Calculus:** How to compute derivatives of matrix expressions
4. **Automatic Differentiation:** How modern deep learning frameworks compute gradients
5. **Optimization:** How gradient-based methods work mathematically
6. **Numerical Stability:** How to write numerically stable code

## Best Practices

### Numerical Stability
- Use appropriate matrix decompositions (e.g., QR instead of normal equations)
- Check condition numbers before solving linear systems
- Prefer stable algorithms (e.g., SVD over eigendecomposition for singular matrices)

### Performance
- Leverage PyTorch's GPU acceleration
- Use batch operations instead of loops
- Understand when to use in-place operations
- Profile code to identify bottlenecks

### Code Quality
- Write readable, documented code
- Use descriptive variable names
- Add assertions to check matrix shapes
- Include error handling for edge cases

## Troubleshooting

### Common Issues

**Issue:** "RuntimeError: CUDA out of memory"
- **Solution:** Reduce batch size or move to CPU with `.cpu()`

**Issue:** "Matrix is singular or near-singular"
- **Solution:** Check condition number, use pseudo-inverse, or add regularization

**Issue:** "Gradients are None"
- **Solution:** Ensure tensors have `requires_grad=True` and operations are differentiable

## Additional Resources

### Online Courses
- [Stanford CS229: Machine Learning - Linear Algebra Review](http://cs229.stanford.edu/section/cs229-linalg.pdf)
- [MIT 18.06: Linear Algebra by Gilbert Strang](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/)

### Books
- "Linear Algebra and Its Applications" by Gilbert Strang
- "Matrix Computations" by Gene Golub and Charles Van Loan
- "Numerical Linear Algebra" by Lloyd N. Trefethen and David Bau III

### PyTorch Documentation
- [PyTorch Linear Algebra Functions](https://pytorch.org/docs/stable/linalg.html)
- [Automatic Differentiation](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)

## Contributing

Contributions are welcome! If you'd like to add:
- Additional decomposition methods
- More practical applications
- Performance optimizations
- Additional visualizations

Please feel free to submit a pull request.

## Citation

If you use these materials in your research or teaching, please cite:

```bibtex
@misc{ai4science_linalg,
  title={Linear Algebra and Matrix Methods for AI4Science},
  author={Your Name},
  year={2025},
  url={https://github.com/nitinnageshkulkarni/AI4Science_Papers_Implementations}
}
```

## License

This project is open source and available under the MIT License.
