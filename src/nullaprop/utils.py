"""
Utility functions for data loading, visualization, and helper functions.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Iterator, Optional, Dict, Any
import matplotlib.pyplot as plt
from functools import partial
from datasets import load_dataset
import jax_dataloader as jdl

def preprocess_mnist_sample(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocess a single MNIST sample.
    
    Args:
        example: Dictionary with 'image' and 'label' keys
        
    Returns:
        Preprocessed example with normalized image
    """
    # Convert PIL image to numpy array
    image = example['image']
    
    # Normalize to [0, 1] and then standardize
    image = image / 255.0
    image = (image - 0.1307) / 0.3081  # MNIST normalization
    
    # Add channel dimension: [H, W] -> [H, W, 1]
    image = np.expand_dims(image, -3)
    
    return {
        'image': image,
        'label': example['label']
    }


def preprocess_cifar10_sample(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocess CIFAR-10 samples (supports batched processing).
    
    Args:
        example: Dictionary with 'img' and 'label' keys
        
    Returns:
        Preprocessed example with normalized image
    """
    # Handle both single samples and batches
    image = example['img']

    # Normalize to [0, 1]
    image = image / 255.0
        
    # CIFAR-10 normalization
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
        
    # Apply normalization
    image = (image - mean) / std
        
    # Change from [H, W, C] to [C, H, W]
    image = np.moveaxis(image, -1, -3)
    
    return {
        'image': image,
        'label': example['label']
    }


def preprocess_cifar100_sample(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocess CIFAR-100 samples (supports batched processing).
    
    Args:
        example: Dictionary with 'img' and 'fine_label' keys
        
    Returns:
        Preprocessed example with normalized image
    """
    # Handle both single samples and batches
    image = example['img']
   
    # Normalize to [0, 1]
    image = image / 255.0
        
    # CIFAR normalization (same as CIFAR-10)
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
        
    # Apply normalization
    image = (image - mean) / std
        
    # Change from [H, W, C] to [C, H, W]
    image = np.moveaxis(image, -1, -3)
    
    return {
        'image': image,
        'label': example['fine_label']  # Use fine-grained labels
    }

def load_data(name, batch_size: int = 512, cache_dir: str = "./data") -> Tuple[Iterator, Iterator]:
    """
    Load dataset using Hugging Face datasets.
    
    Args:
        batch_size: Batch size for data loaders
        cache_dir: Directory to cache downloaded data
        
    Returns:
        Tuple of (train_iterator, test_iterator)
    """
    
    # Load MNIST dataset
    dataset = load_dataset(name, cache_dir=cache_dir).with_format('numpy')
        
    # datasets
    if name == 'mnist':
        hf_train = dataset['train'].map(preprocess_mnist_sample, batched=True, remove_columns=['image', 'label'])
        hf_test = dataset['test'].map(preprocess_mnist_sample, batched=True, remove_columns=['image', 'label'])
    elif name == 'cifar10':
        hf_train = dataset['train'].map(preprocess_cifar10_sample, batched=True, remove_columns=['img', 'label'])
        hf_test = dataset['test'].map(preprocess_cifar10_sample, batched=True, remove_columns=['img', 'label'])
    elif name == 'cifar100':
        hf_train = dataset['train'].map(
            preprocess_cifar100_sample, batched=True, remove_columns=['img', 'fine_label', 'coarse_label'] 
        )
        hf_test = dataset['test'].map(
            preprocess_cifar100_sample, batched=True, remove_columns=['img', 'fine_label', 'coarse_label']
        )
    else:
        raise NotImplementedError

    arr_ds = jdl.ArrayDataset(jnp.asarray(hf_train['image']), jnp.asarray(hf_train['label']))
    train_ds = jdl.DataLoader(
        arr_ds,
        'jax',
        batch_size=batch_size,
        drop_last=True,
        shuffle=True
    )
    
    arr_ds = jdl.ArrayDataset(jnp.asarray(hf_test['image']), jnp.asarray(hf_test['label']))
    test_ds = jdl.DataLoader(
        arr_ds,
        'jax',
        batch_size=10_000,
        shuffle=False
    )

    
    return train_ds, test_ds

def plot_training_curves(losses: list, accuracies: list, title: str = "Training Progress") -> None:
    """
    Plot training loss and accuracy curves.
    
    Args:
        losses: List of loss values per epoch
        accuracies: List of accuracy values per epoch
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = len(losses)
    
    # Plot loss
    ax1.plot(range(1, epochs + 1), losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(range(5, epochs + 1, 5), accuracies, 'o:')
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
    Print a summary of the NoPropCT model parameters.
    
    Args:
        model: NoPropCT model
        input_shape: Shape of input tensor (channels, height, width)
    """
    print("="*60)
    print("NoProp CT Model Summary")
    print("="*60)
    
    print(f"Embedding dimension: {model.embed_dim}")
    print(f"Number of classes: {model.num_classes}")
    print(f"Input shape: {input_shape}")
    
    # Count parameters
    def count_params(pytree):
        return sum(x.size for x in jax.tree_util.tree_leaves(pytree))
    
    cnn_params = count_params(model.cnn)
    label_enc_params = count_params(model.label_enc)
    time_enc_params = count_params(model.time_enc)
    fuse_head_params = count_params(model.fuse_head)
    noise_schedule_params = count_params(model.noise_schedule)
    embed_matrix_params = model.embed_matrix.size
    
    total_params = (cnn_params + label_enc_params + time_enc_params + 
                   fuse_head_params + noise_schedule_params + embed_matrix_params)
    
    print(f"\nParameter counts:")
    print(f"  CNN parameters: {cnn_params:,}")
    print(f"  Label encoder parameters: {label_enc_params:,}")
    print(f"  Time encoder parameters: {time_enc_params:,}")
    print(f"  Fuse head parameters: {fuse_head_params:,}")
    print(f"  Noise schedule parameters: {noise_schedule_params:,}")
    print(f"  Embedding matrix parameters: {embed_matrix_params:,}")
    print(f"  Total parameters: {total_params:,}")
    print("="*60)


def evaluate_model(model, test_iterator, key: jax.random.PRNGKey, num_batches: Optional[int] = None, T_steps: int = 40) -> Tuple[float, float]:
    """
    Evaluate NoPropCT model on test dataset.
    
    Args:
        model: Trained NoPropCT model
        test_iterator: Test data iterator
        key: Random key
        num_batches: Optional limit on number of batches to evaluate
        T_steps: Number of steps for CT inference
        
    Returns:
        Tuple of (accuracy, avg_loss)
    """
    from .training import compute_loss_aligned
    from .inference import inference_ct_heun
    
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    batch_count = 0
    
    for x, y in test_iterator:
        if num_batches is not None and batch_count >= num_batches:
            break
            
        # Split key
        key, subkey_inf, subkey_loss = jax.random.split(key, 3)
        
        # Compute predictions and loss
        predictions = inference_ct_heun(model, x, subkey_inf, T_steps=T_steps)
        loss = compute_loss_aligned(model, x, y, subkey_loss, eta=1.0)
        
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


def initialize_with_prototypes_jax(
    model_cnn_part, # Typically NoPropModel.cnn
    dataset_loader_fn, # A function that returns an iterator, e.g., from load_data
    num_classes: int,
    key: jax.random.PRNGKey,
    samples_per_class: int = 10,
    # embed_dim is inferred from cnn output
) -> jnp.ndarray:
    """
    Initialize W_embed with prototypes from backbone feature space.
    Args:
        model_cnn_part: The CNN part of the NoPropModel.
        dataset_loader_fn: Function that yields batches of (images, labels).
        num_classes: Total number of classes.
        key: JAX random key.
        samples_per_class: Number of samples per class to consider for medoid calculation.
    Returns:
        W_proto: Prototype embeddings [num_classes, cnn_feature_dim].
    """
    print("Initializing prototypes...")
    
    # 1) Embed entire dataset (or a large subset) in batches
    feats_list = []
    labels_list = []
    
    # Assuming model_cnn_part is already a JIT-able function or eqx.Module
    @partial(jax.jit, static_argnums=(2,))
    def get_features_batch(cnn_model, batch_images, feature_dim_static):
        return cnn_model(batch_images)

    for imgs_batch, labels_batch in dataset_loader_fn(): # Expects (images, labels)
        # Ensure images are JAX arrays
        imgs_batch_jax = jnp.asarray(imgs_batch)
        
        # Infer feature_dim from the first batch
        if not feats_list:
            # Temporarily get one feature to know the dimension
            temp_feat_dim = get_features_batch(model_cnn_part, imgs_batch_jax[[0]], -1).shape[-1]
            print(f"Inferred CNN feature dimension for prototypes: {temp_feat_dim}")
        
        feats = get_features_batch(model_cnn_part, imgs_batch_jax, temp_feat_dim)
        feats_list.append(feats)
        labels_list.append(jnp.asarray(labels_batch))

    if not feats_list:
        raise ValueError("Dataset loader did not yield any data for prototype initialization.")

    all_feats = jnp.concatenate(feats_list, axis=0)
    all_labels = jnp.concatenate(labels_list, axis=0)
    
    cnn_feature_dim = all_feats.shape[1]
    W_proto = jnp.zeros((num_classes, cnn_feature_dim))
    
    # 2) For each class, randomly pick samples, then find medoid
    for c in range(num_classes):
        key, subkey = jax.random.split(key)
        
        # Indices of class-c samples
        idxs_c = jnp.where(all_labels == c)[0]
        
        if len(idxs_c) == 0:
            print(f"Warning: No samples found for class {c} during prototype initialization. Using zeros.")
            W_proto = W_proto.at[c].set(jnp.zeros(cnn_feature_dim))
            continue

        # Randomly choose up to samples_per_class
        num_to_sample = min(samples_per_class, len(idxs_c))
        chosen_indices_in_idxs_c = jax.random.choice(subkey, jnp.arange(len(idxs_c)), shape=(num_to_sample,), replace=False)
        chosen_global_indices = idxs_c[chosen_indices_in_idxs_c]
        
        class_embs = all_feats[chosen_global_indices] # [num_to_sample, cnn_feature_dim]
        
        if class_embs.shape[0] == 0: # Should not happen if len(idxs_c) > 0
             print(f"Warning: No embeddings selected for class {c}. Using zeros.")
             W_proto = W_proto.at[c].set(jnp.zeros(cnn_feature_dim))
             continue
        if class_embs.shape[0] == 1: # Only one sample, it's the medoid
            W_proto = W_proto.at[c].set(class_embs[0])
            continue

        # Compute pairwise distances among just these chosen embeddings
        # dmat[i, j] = distance between class_embs[i] and class_embs[j]
        # Using squared Euclidean distance for simplicity, median of distances should still find a central point.
        dmat_sq = jnp.sum((class_embs[:, None, :] - class_embs[None, :, :])**2, axis=-1) # [k, k]
        
        # Sum of distances (or median of distances) to all other points in the selection
        sum_dist_sq = jnp.sum(dmat_sq, axis=1) # [k]
        
        # Medoid is the point with the minimum sum of (squared) distances to others
        best_local_idx = jnp.argmin(sum_dist_sq)
        
        W_proto = W_proto.at[c].set(class_embs[best_local_idx])
        
    print("Prototype initialization complete.")
    return W_proto
