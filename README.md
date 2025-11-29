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

## Getting Started

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0+
- Jupyter Notebook (for interactive tutorials)
- Git for version control

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/nitinnageshkulkarni/AI4Science_Papers_Implementations.git
   cd AI4Science_Papers_Implementations
   ```

2. **Navigate to a specific project:**
   ```bash
   cd Neural_Ordinary_Differential_Equations
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the implementation:**
   ```bash
   # Generate dataset
   python scripts/generate_dataset.py
   
   # Train model
   python scripts/train_neural_ode.py
   ```

### Quick Start Guide

Each project includes:
- 📖 **Comprehensive README** with theory and usage
- 📊 **Dataset generation scripts** for reproducibility
- 🧠 **Training scripts** with best practices
- 📈 **Visualization tools** for results analysis
- 📚 **Reference papers** in the docs folder

## Learning Path

### Recommended Order

1. **Start with Linear Algebra** - Build fundamental understanding
2. **Neural ODEs** - Learn continuous-depth networks
3. **Hamiltonian Neural Networks** - Incorporate physics constraints

### For Different Backgrounds

**Machine Learning Background:**
- Start with Neural ODEs to understand continuous models
- Move to Hamiltonian NNs for physics-informed learning
- Review Linear Algebra as needed for mathematical details

**Physics Background:**
- Start with Hamiltonian Neural Networks (familiar concepts)
- Explore Neural ODEs for general continuous dynamics
- Use Linear Algebra notebooks as PyTorch reference

**Mathematics Background:**
- Begin with Linear Algebra implementations
- Understand theory before moving to applications
- Implement variations and extensions

## Key Features Across Projects

### 🔬 Scientific Rigor
- Implementations based on peer-reviewed papers
- Reproducible results with fixed random seeds
- Comprehensive documentation of methodology

### 💻 Production-Quality Code
- Clean, well-documented Python code
- Type hints and docstrings
- Error handling and validation
- Modular design for reusability

### 📊 Visualization & Analysis
- Publication-quality plots
- Training progress tracking
- Comprehensive evaluation metrics
- Interactive Jupyter notebooks

### 🎓 Educational Focus
- Detailed explanations of concepts
- Step-by-step tutorials
- Common pitfalls and solutions
- References to additional resources

## Repository Statistics

![Stars](https://img.shields.io/github/stars/nitinnageshkulkarni/AI4Science_Papers_Implementations?style=social)
![Forks](https://img.shields.io/github/forks/nitinnageshkulkarni/AI4Science_Papers_Implementations?style=social)
![Issues](https://img.shields.io/github/issues/nitinnageshkulkarni/AI4Science_Papers_Implementations)
![License](https://img.shields.io/github/license/nitinnageshkulkarni/AI4Science_Papers_Implementations)

## Future Additions

Planned implementations include:

- [ ] **Lagrangian Neural Networks** - Learning from Lagrangian mechanics
- [ ] **Fourier Neural Operators** - Learning operators for PDEs
- [ ] **Graph Neural Networks** - Message passing on graphs
- [ ] **Physics-Informed Neural Networks (PINNs)** - PDE constraints
- [ ] **Diffusion Models** - Score-based generative models
- [ ] **Neural Operators** - Operator learning frameworks

## Contributing

Contributions are welcome! Here's how you can help:

### Ways to Contribute

1. **Implement new papers** following the project structure
2. **Improve documentation** with clearer explanations
3. **Add visualizations** for better understanding
4. **Report bugs** or suggest enhancements
5. **Share your results** from using these implementations

### Contribution Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please ensure:
- Code follows PEP 8 style guidelines
- Documentation is comprehensive
- Tests pass (if applicable)
- Results are reproducible

## Citation

If you use this repository in your research or teaching, please cite:

```bibtex
@misc{ai4science_implementations,
  title={AI4Science Papers Implementations},
  author={Nitin Nagesh Kulkarni},
  year={2025},
  publisher={GitHub},
  url={https://github.com/nitinnageshkulkarni/AI4Science_Papers_Implementations}
}
```

## Resources

### Online Communities
- [AI4Science Discord](https://discord.gg/ai4science)
- [r/MachineLearning](https://www.reddit.com/r/MachineLearning/)
- [Papers with Code](https://paperswithcode.com/)

### Related Repositories
- [PyTorch Examples](https://github.com/pytorch/examples)
- [Deep Learning Papers](https://github.com/terryum/awesome-deep-learning-papers)
- [Scientific Machine Learning](https://github.com/SciML)

### Conferences & Journals
- NeurIPS (Neural Information Processing Systems)
- ICML (International Conference on Machine Learning)
- ICLR (International Conference on Learning Representations)
- Nature Machine Intelligence
- Journal of Machine Learning Research

## Acknowledgments

This project is inspired by and built upon the groundbreaking work of researchers in the AI4Science community. Special thanks to:

- The authors of all implemented papers
- PyTorch and scientific Python communities
- Open-source contributors and maintainers

## Contact

**Nitin Nagesh Kulkarni**
- GitHub: [@nitinnageshkulkarni](https://github.com/nitinnageshkulkarni)
- Repository: [AI4Science_Papers_Implementations](https://github.com/nitinnageshkulkarni/AI4Science_Papers_Implementations)

## License

This project is open source and available under the MIT License.

---

⭐ **Star this repository** if you find it helpful!

🔔 **Watch** for updates on new implementations!

🍴 **Fork** to create your own implementations!
