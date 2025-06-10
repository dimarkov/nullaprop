# NoProp: JAX/Equinox Implementation

A JAX/Equinox implementation of the **NoProp** algorithm, inspired by the continuous-time formulation presented in ["Noisy Label Learning with Diffusion Models"](https://arxiv.org/abs/2302.13040) (referred to as NoPropCT in the `yhgon/NoProp` repository) and the original NoProp paper. This version focuses on a continuous-time diffusion process for label embeddings.

## 🎯 What is NoProp?

NoProp (No Propagation) is a training paradigm. This implementation adapts concepts from diffusion models for robust learning, particularly in the context of label embeddings. It draws inspiration from the PyTorch implementation by `yhgon/NoProp` which explores a continuous-time (CT) variant.

### Key Concepts in this Implementation
- **Continuous-Time Diffusion**: Models the evolution of label embeddings `z_t` over a continuous time `t` from 0 (noisy) to 1 (cleaner).
- **Score Matching Ideas**: The loss function incorporates terms related to predicting clean embeddings from noisy ones, guided by a noise schedule.
- **Component-Based Architecture**: Uses distinct modules for image feature extraction, label encoding, time encoding, and fusing these features to make predictions.
- **Prototype-Based Embeddings**: Optionally allows initialization of class embeddings using medoids from the feature space.

## 🚀 JAX Implementation Features

This JAX/Equinox implementation offers:

- ✅ **Modular Design**: Clear separation of components like `CNN`, `LabelEncoder`, `TimeEncoder`, `FuseHead`, and `NoiseSchedule`.
- ✅ **JIT Compilation**: Leverages JAX for optimized execution of training and inference steps.
- ✅ **Equinox for Models**: Utilizes Equinox for elegant and Pythonic model definitions.
- ✅ **Inspired by `yhgon/NoProp`**: Aims to translate the core ideas of the continuous-time PyTorch NoProp variant into JAX.
- ✅ **HuggingFace Datasets**: Uses `datasets` library for data loading.
- ✅ **Flexible Embedding Initialization**: Supports one-hot, learnable, and prototype-based class embeddings.

## 📦 Installation

```bash
git clone git@github.com:dimarkov/nullaprop.git
cd nullaprop
pip install -e .
```

## 🏃‍♂️ Quick Start

### Basic Usage (Illustrative - API will match new model)
```python
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from nullaprop.models import init_noprop_model 
from nullaprop.training import create_train_state, train_epoch 
from nullaprop.inference import inference_ct_euler
from nullaprop.utils import load_data, initialize_with_prototypes_jax

# --- Configuration ---
key = jax.random.PRNGKey(42)
num_classes = 10
embed_dim = 64  # Dimension for label, time, and image feature embeddings
time_emb_dim_internal = 64 # Raw sinusoidal embedding dimension
input_channels = 1 # MNIST
epochs = 10
batch_size = 128
learning_rate = 1e-3
eta = 1.0 # Loss hyperparameter
prototype = False # use prototype based initialisation of embeding matrix

# --- Initialize Model ---
key, model_key = jax.random.split(key)
model = init_noprop_model(
    key=model_key,
    num_classes=num_classes,
    embed_dim=embed_dim,
    input_channels=input_channels,
    time_emb_dim_internal=time_emb_dim_internal,
)

# --- Load Data ---
# train_loader_fn is a function that returns an iterator
train_loader, test_loader = load_data("mnist", batch_size=batch_size)

# --- Prototype Initialization (Optional) ---
if prototype: # Check attribute on model instance
    key, proto_key = jax.random.split(key)
    # Ensure the dataset_loader_fn for prototypes provides enough diverse samples
    # For MNIST, the full training set is usually fine for prototype init.
    # The feature_dim of prototypes will match model.cnn.linear.out_features (which is embed_dim)
    prototypes = initialize_with_prototypes_jax(
        model_cnn_part=model.cnn,
        dataset_loader_fn=train_loader_fn, # Use training data
        num_classes=num_classes,
        key=proto_key,
        samples_per_class=20 # More samples for better medoids
    )
    # Update model's embed_matrix.
    model = eqx.tree_at(lambda m: m.embed_matrix, model, prototypes)


# --- Training Setup ---
key, state_key = jax.random.split(key)
optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=1e-3)
state = create_train_state(model, optimizer, key=state_key, learning_rate=learning_rate)

# --- Training Loop ---
for epoch in range(1, epochs + 1):
    # Pass the optimizer explicitly if it's external to create_train_state/train_step
    state, avg_loss = train_epoch(state, train_loader, optimizer, eta=eta)
    print(f"Epoch {epoch:03d} Avg Loss: {avg_loss:.4f}")

    # Evaluation (example)
    if epoch % 5 == 0:
        eval_key, state_key_new = jax.random.split(state.key) # Consume key from state
        
        test_images, test_labels = next(test_loader)
        
        # Use the correct inference function
        preds = inference_ct_euler(state.model, test_images, eval_key, T_steps=40)
        acc = jnp.mean(preds == test_labels)
        print(f"Epoch {epoch:03d} Test Accuracy (batch): {acc:.4f}")
        state = state._replace(key=state_key_new) # Update key in state

print("Training complete.")

# --- Inference Example ---
infer_key, state_key_final = jax.random.split(state.key)
sample_images, _ = next(test_loader)
predictions = inference_ct_euler(state.model, sample_images, infer_key, T_steps=100)
print(f"Sample predictions: {predictions}")

```

## 🧪 Experiments and Demos

### Interactive Demo
```bash
jupyter notebook demo_noprop.ipynb
```

### Command Line Experiments
```python
# MNIST experiment
nop.run_experiment('mnist', epochs=50)

# CIFAR-10 experiment  
nop.run_experiment('cifar10', epochs=150)

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
3. **`nullaprop/inference.py`**: Stochastic inference
4. **`nullaprop/utils.py`**: Data loading, visualization, and utilities
5. **`nullaprop/experiments.py`**: Pre-configured experiments and benchmarks

### Model Structure
```
NoPropModel
├── cnn: CNN feature extractor (outputs features of `embed_dim`)
├── label_enc: LabelEncoder (processes noisy label embeddings)
├── time_enc: TimeEncoder (processes continuous time `t`)
├── fuse_head: FuseHead (combines features from cnn, label_enc, time_enc, and outputs class logits)
├── noise_schedule: NoiseSchedule (provides ᾱ(t) based on learnable or fixed schedule)
└── embed_matrix: W_embed (target class embeddings [num_classes, embed_dim])
```

## ⚡ Key Operations

### Training (`compute_loss_aligned`)
The loss function combines three main components:
1.  **`loss_ce`**: Cross-entropy loss. Evaluates the model's ability to classify given a noisy label embedding at `t=1`.
2.  **`loss_kl`**: KL divergence term. Regularizes the clean label embeddings `u_y` (from `model.embed_matrix`).
3.  **`loss_sdm`**: Score diffusion matching-like term. Penalizes the difference between the model's predicted clean embedding `pred_e` and the true clean embedding `u_y`, weighted by the derivative of the Signal-to-Noise Ratio (SNR), `snr_p`. `pred_e` is obtained by `softmax(logits) @ model.embed_matrix`.

### Inference (`inference_ct_euler` / `inference_ct_heun`)
Inference is performed by solving a reverse-time ordinary differential equation (ODE) using methods like Euler or Heun.
- Starts with random noise `z_0 ~ N(0,I)`.
- Iteratively refines `z_t` towards `t=1` using the model's predictions:
  `dz/dt = (pred_e(z,t) - z) / (1 - ᾱ(t))`
  where `pred_e(z,t)` is the model's estimate of the clean embedding given noisy `z` at time `t`.
- The final logits are obtained by feeding the evolved `z` at `t=1` back into the model.

<!-- ## 📈 Benchmarks (To be updated for the new model)

Performance comparison on MNIST (T=10, batch_size=128):

| Metric | NoProp-JAX | Traditional BP |
|--------|------------|----------------|
| Training Time | ~0.5s/epoch | ~0.8s/epoch |
| Memory Usage | 0.49 GB | 0.87 GB |
| Parallelizable | ✅ Yes | ❌ No |
| Final Accuracy | 99.4% | 99.5% | -->


## 📄 Citation

If you use this implementation, please cite the relevant papers and this repository:

```bibtex
@article{li2023noisy,
  title={Noisy Label Learning with Diffusion Models},
  author={Li, Qinyu and Shen, Yilun and Zhao, Zhaowei and Liu, Shurui and Liu, Bing},
  journal={arXiv preprint arXiv:2302.13040},
  year={2023}
}

@article{noprop_original_paper,
  title={NoProp: Training Neural Networks Without Back-propagation or Forward-propagation},
  author={Li, Qinyu and Teh, Yee Whye and Pascanu, Razvan},
  journal={arXiv preprint arXiv:2403.04322},
  year={2024}
}

@software{nullaprop2025,
  title={nullaprop: JAX/Equinox Implementation of Continuous-Time NoProp},
  author={Dimitrije Markovic},
  year={2025},
  url={https://github.com/dimarkov/nullaprop}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- Original NoProp Paper: [arXiv:2403.04322](https://arxiv.org/abs/2403.04322)
- Noisy Label Learning with Diffusion Models: [arXiv:2302.13040](https://arxiv.org/abs/2302.13040)
- `yhgon/NoProp` PyTorch repository (inspiration for this JAX port): [https://github.com/yhgon/NoProp](https://github.com/yhgon/NoProp)
- [JAX Documentation](https://jax.readthedocs.io/)
- [Equinox Documentation](https://docs.kidger.site/equinox/)
- [Demo Notebook](demo_noprop.ipynb)

---
