# NoProp: JAX/Equinox Implementation

A high-performance JAX/Equinox implementation of the **NoProp** algorithm - a neural network training method that eliminates both back-propagation and forward-propagation.

## 🎯 What is NoProp?

NoProp (No Propagation) is a novel training paradigm introduced in the paper ["NoProp: Training Neural Networks Without Back-propagation or Forward-propagation"](https://github.com/Sid3503/NoProp). Unlike traditional deep learning methods that rely on gradient back-propagation, NoProp trains each layer independently using a diffusion-based denoising process.

### Key Innovation
- **No Back-propagation**: Each layer learns independently without gradient flow
- **No Forward-propagation**: Training doesn't require sequential layer computation  
- **Diffusion-based**: Uses denoising score matching from diffusion models
- **Parallelizable**: All layers can be trained simultaneously

## 🚀 JAX Implementation Features

Our JAX/Equinox implementation provides significant improvements over traditional frameworks:

- ✅ **Parallelized computation** using JAX `vmap` over layer parameters
- ✅ **Memory-efficient diffusion** using `jax.lax.scan`
- ✅ **JIT compilation** for optimized execution
- ✅ **Pure functional programming** with Equinox
- ✅ **Multi-device support** via JAX's device parallelism
- ✅ **No PyTorch dependency** - uses HuggingFace datasets
- ✅ **Comprehensive benchmarking** and visualization tools

## 📦 Installation

### Prerequisites
```bash
pip install jax jaxlib equinox optax datasets matplotlib numpy
```

### Clone and Install
```bash
git clone git@github.com:dimarkov/nullaprop.git
cd nullaprop
pip install -e .
```

## 🏃‍♂️ Quick Start

### Basic Usage
```python
import nullaprop as nop

# Run MNIST experiment
results = nop.run_mnist_experiment(
    epochs=50,
    batch_size=128,
    learning_rate=1e-3,
    T=10,  # diffusion steps
    seed=42
)

print(f"Final accuracy: {results['final_accuracy']:.4f}")
```

### Manual Model Creation
```python
import jax
import nullaprop as nop

# Initialize model
key = jax.random.PRNGKey(42)
model = nop.init_noprop_model(
    key=key,
    T=10,                    # diffusion steps
    embed_dim=10,            # number of classes
    feature_dim=128,         # hidden dimension
    input_channels=1         # grayscale images
)

# Load data and train
train_iter, test_iter = nop.load_mnist_data(batch_size=128)
state = nop.create_train_state(model, learning_rate=1e-3)

# Training step
for x, y in train_iter():
    key, subkey = jax.random.split(key)
    state, loss = nop.train_step(state, x, y, subkey, optimizer)
    break

# Inference
predictions = nop.inference_step(state.model, x, key)
```


## 🧪 Experiments and Demos

### Interactive Demo
```bash
jupyter notebook demo_noprop.ipynb
```

### Command Line Experiments
```python
# MNIST experiment
nop.run_mnist_experiment(epochs=50)

# CIFAR-10 experiment  
nop.run_cifar10_experiment(epochs=150)

# Visualize diffusion process
nop.demonstrate_diffusion_process("mnist")

# Benchmark performance
results = nop.benchmark_performance(
    dataset="mnist",
    batch_sizes=[32, 64, 128],
    T_values=[5, 10, 20]
)
```

## 🏗️ Architecture

### Core Components

1. **`nullaprop/models.py`**: Model definitions using Equinox
2. **`nullaprop/training.py`**: Training logic with parallelized layer updates
3. **`nullaprop/inference.py`**: Stochastic and deterministic inference
4. **`nullaprop/utils.py`**: Data loading, visualization, and utilities
5. **`nullaprop/experiments.py`**: Pre-configured experiments and benchmarks

### Model Structure
```
NoPropModel
├── cnn: CNN feature extractor (Equinox)
├── mlp_params: Vectorized MLP parameters [T, param_size]  
├── alpha_schedule: Noise schedule [T]
├── embed_matrix: Class embeddings [num_classes, embed_dim]
└── classifier: Final classification layer
```

## ⚡ Performance Optimizations

### Parallelization Strategy
- **vmap over layers**: All T diffusion layers computed in parallel
- **scan for sequences**: Memory-efficient diffusion process simulation
- **JIT compilation**: Entire computation graph optimized
- **Device parallelism**: Easy multi-GPU/TPU scaling

### Memory Efficiency
- Scan-based diffusion sequences prevent memory explosion
- Functional programming eliminates mutable state overhead
- Optional gradient checkpointing for large models

<!-- ## 📈 Benchmarks

Performance comparison on MNIST (T=10, batch_size=128):

| Metric | NoProp-JAX | Traditional BP |
|--------|------------|----------------|
| Training Time | ~0.5s/epoch | ~0.8s/epoch |
| Memory Usage | 0.49 GB | 0.87 GB |
| Parallelizable | ✅ Yes | ❌ No |
| Final Accuracy | 99.4% | 99.5% | -->


## 📄 Citation

If you use this implementation, please cite both the original paper and this repository:

```bibtex
@article{noprop2024,
  title={NoProp: Training Neural Networks Without Back-propagation or Forward-propagation},
  author={Li, Qinyu and Teh, Yee Whye and Pascanu, Razvan},
  journal={arXiv preprint arXiv:2503.24322},
  year={2024}
}

@software{nullaprop2025,
  title={nullaprop: JAX/Equinox Implementation},
  author={Dimitrije Markovic},
  year={2025},
  url={https://github.com/dimarkov/nullaprop}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- [Original Paper](https://github.com/Sid3503/NoProp/blob/main/NoProp.pdf)
- [JAX Documentation](https://jax.readthedocs.io/)
- [Equinox Documentation](https://docs.kidger.site/equinox/)
- [Demo Notebook](demo_noprop.ipynb)

---
