"""
Utility functions for data loading, visualization, and helper functions.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Iterator, Optional, Dict, Any
import matplotlib.pyplot as plt
from functools import partial

try:
    from datasets import load_dataset
    import datasets
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Hugging Face datasets not available. Install with: pip install datasets")


def create_noise_schedule(T: int, schedule_type: str = "linear") -> jnp.ndarray:
    """
    Create noise schedule for diffusion process.
    
    Args:
        T: Number of time steps
        schedule_type: Type of schedule ("linear", "cosine")
        
    Returns:
        Alpha schedule [T]
    """
    if schedule_type == "linear":
        return jnp.linspace(1.0, 0.1, T)
    elif schedule_type == "cosine":
        # Cosine schedule as used in DDPM
        s = 0.008
        steps = jnp.arange(T + 1) / T
        alphas_cumprod = jnp.cos((steps + s) / (1 + s) * jnp.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        alphas = jnp.concatenate([jnp.array([1.0]), alphas_cumprod[1:] / alphas_cumprod[:-1]])
        return jnp.clip(alphas[1:], 0.0001, 0.9999)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


def preprocess_mnist_sample(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocess a single MNIST sample.
    
    Args:
        example: Dictionary with 'image' and 'label' keys
        
    Returns:
        Preprocessed example with normalized image
    """
    # Convert PIL image to numpy array
    image = np.array(example['image'], dtype=np.float32)
    
    # Normalize to [0, 1] and then standardize
    image = image / 255.0
    image = (image - 0.1307) / 0.3081  # MNIST normalization
    
    # Add channel dimension: [H, W] -> [1, H, W]
    image = image[None, :, :]
    
    return {
        'image': image,
        'label': example['label']
    }


def preprocess_cifar10_sample(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocess a single CIFAR-10 sample.
    
    Args:
        example: Dictionary with 'img' and 'label' keys
        
    Returns:
        Preprocessed example with normalized image
    """
    # Convert PIL image to numpy array
    image = np.array(example['img'], dtype=np.float32)
    
    # Normalize to [0, 1]
    image = image / 255.0
    
    # CIFAR-10 normalization
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    
    # Apply normalization
    image = (image - mean) / std
    
    # Change from [H, W, C] to [C, H, W]
    image = np.transpose(image, (2, 0, 1))
    
    return {
        'image': image,
        'label': example['label']
    }


def load_mnist_data(batch_size: int = 128, cache_dir: str = "./data") -> Tuple[Iterator, Iterator]:
    """
    Load MNIST dataset using Hugging Face datasets.
    
    Args:
        batch_size: Batch size for data loaders
        cache_dir: Directory to cache downloaded data
        
    Returns:
        Tuple of (train_iterator, test_iterator)
    """
    if not DATASETS_AVAILABLE:
        raise ImportError("Hugging Face datasets is required. Install with: pip install datasets")
    
    # Load MNIST dataset
    dataset = load_dataset("mnist", cache_dir=cache_dir)
    
    # Preprocess datasets
    train_dataset = dataset['train'].map(preprocess_mnist_sample, remove_columns=['image', 'label'])
    test_dataset = dataset['test'].map(preprocess_mnist_sample, remove_columns=['image', 'label'])
    
    # Set format to numpy for JAX compatibility
    train_dataset.set_format(type='numpy', columns=['image', 'label'])
    test_dataset.set_format(type='numpy', columns=['image', 'label'])
    
    # Create iterators
    def create_iterator(dataset, batch_size, shuffle=True):
        """Create a data iterator."""
        if shuffle:
            dataset = dataset.shuffle(seed=42)
        
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            # Ensure we have a full batch for training
            if len(batch['image']) == batch_size or not shuffle:
                yield jnp.array(batch['image']), jnp.array(batch['label'])
    
    train_iterator = lambda: create_iterator(train_dataset, batch_size, shuffle=True)
    test_iterator = lambda: create_iterator(test_dataset, batch_size, shuffle=False)
    
    return train_iterator, test_iterator


def load_cifar10_data(batch_size: int = 128, cache_dir: str = "./data") -> Tuple[Iterator, Iterator]:
    """
    Load CIFAR-10 dataset using Hugging Face datasets.
    
    Args:
        batch_size: Batch size for data loaders
        cache_dir: Directory to cache downloaded data
        
    Returns:
        Tuple of (train_iterator, test_iterator)
    """
    if not DATASETS_AVAILABLE:
        raise ImportError("Hugging Face datasets is required. Install with: pip install datasets")
    
    # Load CIFAR-10 dataset
    dataset = load_dataset("cifar10", cache_dir=cache_dir)
    
    # Preprocess datasets
    train_dataset = dataset['train'].map(preprocess_cifar10_sample, remove_columns=['img', 'label'])
    test_dataset = dataset['test'].map(preprocess_cifar10_sample, remove_columns=['img', 'label'])
    
    # Set format to numpy
    train_dataset.set_format(type='numpy', columns=['image', 'label'])
    test_dataset.set_format(type='numpy', columns=['image', 'label'])
    
    # Create iterators
    def create_iterator(dataset, batch_size, shuffle=True):
        """Create a data iterator."""
        if shuffle:
            dataset = dataset.shuffle(seed=42)
        
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            if len(batch['image']) == batch_size or not shuffle:
                yield jnp.array(batch['image']), jnp.array(batch['label'])
    
    train_iterator = lambda: create_iterator(train_dataset, batch_size, shuffle=True)
    test_iterator = lambda: create_iterator(test_dataset, batch_size, shuffle=False)
    
    return train_iterator, test_iterator


def load_cifar100_data(batch_size: int = 128, cache_dir: str = "./data") -> Tuple[Iterator, Iterator]:
    """
    Load CIFAR-100 dataset using Hugging Face datasets.
    
    Args:
        batch_size: Batch size for data loaders
        cache_dir: Directory to cache downloaded data
        
    Returns:
        Tuple of (train_iterator, test_iterator)
    """
    if not DATASETS_AVAILABLE:
        raise ImportError("Hugging Face datasets is required. Install with: pip install datasets")
    
    # Load CIFAR-100 dataset
    dataset = load_dataset("cifar100", cache_dir=cache_dir)
    
    # Use fine labels (100 classes)
    def preprocess_cifar100_sample(example):
        # Similar to CIFAR-10 but use fine_label
        image = np.array(example['img'], dtype=np.float32) / 255.0
        
        # CIFAR normalization (same as CIFAR-10)
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2023, 0.1994, 0.2010])
        image = (image - mean) / std
        image = np.transpose(image, (2, 0, 1))
        
        return {
            'image': image,
            'label': example['fine_label']  # Use fine-grained labels
        }
    
    # Preprocess datasets
    train_dataset = dataset['train'].map(preprocess_cifar100_sample, remove_columns=['img', 'fine_label', 'coarse_label'])
    test_dataset = dataset['test'].map(preprocess_cifar100_sample, remove_columns=['img', 'fine_label', 'coarse_label'])
    
    # Set format to numpy
    train_dataset.set_format(type='numpy', columns=['image', 'label'])
    test_dataset.set_format(type='numpy', columns=['image', 'label'])
    
    # Create iterators
    def create_iterator(dataset, batch_size, shuffle=True):
        if shuffle:
            dataset = dataset.shuffle(seed=42)
        
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            if len(batch['image']) == batch_size or not shuffle:
                yield jnp.array(batch['image']), jnp.array(batch['label'])
    
    train_iterator = lambda: create_iterator(train_dataset, batch_size, shuffle=True)
    test_iterator = lambda: create_iterator(test_dataset, batch_size, shuffle=False)
    
    return train_iterator, test_iterator


def visualize_diffusion_process(clean_labels: jnp.ndarray, 
                               noisy_sequence: jnp.ndarray,
                               alpha_schedule: jnp.ndarray,
                               class_names: Optional[list] = None) -> None:
    """
    Visualize the diffusion process for label corruption.
    
    Args:
        clean_labels: Original one-hot labels [batch_size, num_classes]
        noisy_sequence: Sequence of noisy labels [T, batch_size, num_classes]
        alpha_schedule: Alpha values [T]
        class_names: Optional list of class names
    """
    T, batch_size, num_classes = noisy_sequence.shape
    
    # Select first sample for visualization
    sample_idx = 0
    clean_sample = clean_labels[sample_idx]
    noisy_samples = noisy_sequence[:, sample_idx, :]
    
    # Create figure
    fig, axes = plt.subplots(2, T//2 + 1, figsize=(15, 6))
    axes = axes.flatten()
    
    # Plot clean label
    axes[0].bar(range(num_classes), clean_sample)
    axes[0].set_title('Clean Label (t=0)')
    axes[0].set_ylim([0, 1])
    if class_names:
        axes[0].set_xticks(range(num_classes))
        axes[0].set_xticklabels(class_names, rotation=45)
    
    # Plot noisy versions
    for t in range(min(T, len(axes)-1)):
        ax_idx = t + 1
        if ax_idx < len(axes):
            axes[ax_idx].bar(range(num_classes), noisy_samples[t])
            axes[ax_idx].set_title(f'Noisy t={t+1} (α={alpha_schedule[t]:.2f})')
            axes[ax_idx].set_ylim([0, 1])
            if class_names:
                axes[ax_idx].set_xticks(range(num_classes))
                axes[ax_idx].set_xticklabels(class_names, rotation=45)
    
    # Remove unused subplots
    for ax_idx in range(T+1, len(axes)):
        fig.delaxes(axes[ax_idx])
    
    plt.tight_layout()
    plt.show()


def visualize_inference_process(predictions: jnp.ndarray,
                               intermediate_states: jnp.ndarray,
                               true_label: int,
                               class_names: Optional[list] = None) -> None:
    """
    Visualize the inference (reverse diffusion) process.
    
    Args:
        predictions: Final predictions [batch_size]
        intermediate_states: Intermediate states [T, batch_size, num_classes]
        true_label: True class label
        class_names: Optional list of class names
    """
    T, batch_size, num_classes = intermediate_states.shape
    
    # Select first sample
    sample_idx = 0
    sample_states = intermediate_states[:, sample_idx, :]
    final_pred = predictions[sample_idx]
    
    # Create figure
    fig, axes = plt.subplots(2, T//2 + 1, figsize=(15, 6))
    axes = axes.flatten()
    
    # Plot intermediate states
    for t in range(min(T, len(axes))):
        axes[t].bar(range(num_classes), sample_states[t])
        axes[t].set_title(f'Reverse t={T-t-1}')
        axes[t].set_ylim([sample_states.min(), sample_states.max()])
        if class_names:
            axes[t].set_xticks(range(num_classes))
            axes[t].set_xticklabels(class_names, rotation=45)
    
    # Add final prediction info
    if len(axes) > T:
        axes[T].text(0.1, 0.5, f'True: {true_label}\nPred: {final_pred}\nCorrect: {true_label == final_pred}', 
                    fontsize=12, transform=axes[T].transAxes)
        axes[T].set_xlim([0, 1])
        axes[T].set_ylim([0, 1])
        axes[T].axis('off')
    
    # Remove unused subplots
    for ax_idx in range(T+1, len(axes)):
        fig.delaxes(axes[ax_idx])
    
    plt.tight_layout()
    plt.show()


def plot_training_curves(losses: list, accuracies: list, title: str = "Training Progress") -> None:
    """
    Plot training loss and accuracy curves.
    
    Args:
        losses: List of loss values per epoch
        accuracies: List of accuracy values per epoch
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(accuracies)
    ax2.set_title('Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def compute_class_distribution(labels: jnp.ndarray, num_classes: int) -> jnp.ndarray:
    """
    Compute class distribution from labels.
    
    Args:
        labels: Label array [num_samples]
        num_classes: Number of classes
        
    Returns:
        Class counts [num_classes]
    """
    return jnp.bincount(labels, length=num_classes)


def print_model_summary(model, input_shape: Tuple[int, ...]) -> None:
    """
    Print a summary of the model parameters.
    
    Args:
        model: NoProp model
        input_shape: Shape of input tensor (channels, height, width)
    """
    print("="*60)
    print("NoProp Model Summary")
    print("="*60)
    
    print(f"Diffusion steps (T): {model.T}")
    print(f"Embedding dimension: {model.embed_dim}")
    print(f"Feature dimension: {model.feature_dim}")
    print(f"Input shape: {input_shape}")
    
    # Count parameters
    def count_params(pytree):
        return sum(x.size for x in jax.tree_util.tree_leaves(pytree))
    
    cnn_params = count_params(model.cnn)
    mlp_params = count_params(model.mlp_params)
    total_params = cnn_params + mlp_params
    
    print(f"\nParameter counts:")
    print(f"  CNN parameters: {cnn_params:,}")
    print(f"  MLP parameters: {mlp_params:,}")
    print(f"  Total parameters: {total_params:,}")
    
    print(f"\nAlpha schedule: {model.alpha_schedule}")
    print("="*60)


def evaluate_model(model, test_iterator, key: jax.random.PRNGKey, num_batches: Optional[int] = None) -> Tuple[float, float]:
    """
    Evaluate model on test dataset.
    
    Args:
        model: Trained NoProp model
        test_iterator: Test data iterator
        key: Random key
        num_batches: Optional limit on number of batches to evaluate
        
    Returns:
        Tuple of (accuracy, avg_loss)
    """
    from .training import compute_loss
    from .inference import inference_step
    
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    batch_count = 0
    
    for x, y in test_iterator():
        if num_batches is not None and batch_count >= num_batches:
            break
            
        # Split key
        key, subkey = jax.random.split(key)
        
        # Compute predictions and loss
        predictions = inference_step(model, x, subkey)
        loss = compute_loss(model, x, y, subkey)
        
        # Update statistics
        correct = jnp.sum(predictions == y)
        total_correct += correct
        total_samples += len(y)
        total_loss += loss
        batch_count += 1
    
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
    
    return float(accuracy), float(avg_loss)


def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """
    Get information about supported datasets.
    
    Args:
        dataset_name: Name of dataset ("mnist", "cifar10", "cifar100")
        
    Returns:
        Dictionary with dataset information
    """
    dataset_info = {
        "mnist": {
            "num_classes": 10,
            "input_channels": 1,
            "input_size": (28, 28),
            "class_names": [str(i) for i in range(10)]
        },
        "cifar10": {
            "num_classes": 10,
            "input_channels": 3,
            "input_size": (32, 32),
            "class_names": ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                           'dog', 'frog', 'horse', 'ship', 'truck']
        },
        "cifar100": {
            "num_classes": 100,
            "input_channels": 3,
            "input_size": (32, 32),
            "class_names": None  # Too many to list here
        }
    }
    
    if dataset_name not in dataset_info:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {list(dataset_info.keys())}")
    
    return dataset_info[dataset_name]
