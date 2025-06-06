"""
Experiments and benchmarks for NoProp JAX implementation.
"""

import jax
import jax.numpy as jnp
import optax
import time
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt

from .models import init_noprop_model
from .training import create_train_state, train_step, compute_loss, compute_accuracy
from .inference import inference_step, inference_with_intermediate
from .utils import (
    load_mnist_data, load_cifar10_data, load_cifar100_data,
    get_dataset_info, print_model_summary, evaluate_model,
    plot_training_curves, visualize_diffusion_process,
    create_noise_schedule
)


def run_mnist_experiment(
    epochs: int = 50,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    T: int = 10,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Run NoProp experiment on MNIST dataset.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        T: Number of diffusion steps
        seed: Random seed
        
    Returns:
        Dictionary with experiment results
    """
    print("="*60)
    print("NOPROP MNIST EXPERIMENT")
    print("="*60)
    
    # Initialize random key
    key = jax.random.PRNGKey(seed)
    key, model_key = jax.random.split(key)
    
    # Load MNIST data
    print("Loading MNIST dataset...")
    train_iterator, test_iterator = load_mnist_data(batch_size=batch_size)
    dataset_info = get_dataset_info("mnist")
    
    # Initialize model
    print("Initializing NoProp model...")
    model = init_noprop_model(
        key=model_key,
        T=T,
        embed_dim=dataset_info["num_classes"],
        feature_dim=128,
        input_channels=dataset_info["input_channels"]
    )
    
    # Print model summary
    print_model_summary(model, dataset_info["input_size"])
    
    # Create training state
    optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=1e-4)
    state = create_train_state(model, learning_rate=learning_rate)
    
    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    losses = []
    accuracies = []
    
    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        num_batches = 0
        
        # Training
        for x, y in train_iterator():
            key, subkey = jax.random.split(key)
            state, loss = train_step(state, x, y, subkey, optimizer)
            epoch_loss += loss
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        losses.append(float(avg_loss))
        
        # Evaluation every 5 epochs
        if epoch % 5 == 0 or epoch == epochs - 1:
            key, eval_key = jax.random.split(key)
            accuracy, eval_loss = evaluate_model(state.model, test_iterator, eval_key, num_batches=10)
            accuracies.append(accuracy)
            
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | Test Acc: {accuracy:.4f} | Time: {epoch_time:.2f}s")
        else:
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")
    
    # Final evaluation
    print("\nFinal evaluation...")
    key, eval_key = jax.random.split(key)
    final_accuracy, final_loss = evaluate_model(state.model, test_iterator, eval_key)
    
    print(f"Final Test Accuracy: {final_accuracy:.4f}")
    print(f"Final Test Loss: {final_loss:.4f}")
    
    # Plot training curves
    plot_training_curves(losses, accuracies, "NoProp MNIST Training")
    
    return {
        "model": state.model,
        "final_accuracy": final_accuracy,
        "final_loss": final_loss,
        "losses": losses,
        "accuracies": accuracies,
        "dataset": "mnist"
    }


def run_cifar10_experiment(
    epochs: int = 150,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    T: int = 10,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Run NoProp experiment on CIFAR-10 dataset.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        T: Number of diffusion steps
        seed: Random seed
        
    Returns:
        Dictionary with experiment results
    """
    print("="*60)
    print("NOPROP CIFAR-10 EXPERIMENT")
    print("="*60)
    
    # Initialize random key
    key = jax.random.PRNGKey(seed)
    key, model_key = jax.random.split(key)
    
    # Load CIFAR-10 data
    print("Loading CIFAR-10 dataset...")
    train_iterator, test_iterator = load_cifar10_data(batch_size=batch_size)
    dataset_info = get_dataset_info("cifar10")
    
    # Initialize model
    print("Initializing NoProp model...")
    model = init_noprop_model(
        key=model_key,
        T=T,
        embed_dim=dataset_info["num_classes"],
        feature_dim=128,
        input_channels=dataset_info["input_channels"]
    )
    
    # Print model summary
    print_model_summary(model, dataset_info["input_size"])
    
    # Create training state with learning rate schedule
    schedule = optax.exponential_decay(learning_rate, transition_steps=1000, decay_rate=0.95)
    optimizer = optax.adamw(learning_rate=schedule, weight_decay=1e-4)
    state = create_train_state(model, learning_rate=learning_rate)
    
    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    losses = []
    accuracies = []
    
    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        num_batches = 0
        
        # Training
        for x, y in train_iterator():
            key, subkey = jax.random.split(key)
            state, loss = train_step(state, x, y, subkey, optimizer)
            epoch_loss += loss
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        losses.append(float(avg_loss))
        
        # Evaluation every 10 epochs
        if epoch % 10 == 0 or epoch == epochs - 1:
            key, eval_key = jax.random.split(key)
            accuracy, eval_loss = evaluate_model(state.model, test_iterator, eval_key, num_batches=10)
            accuracies.append(accuracy)
            
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | Test Acc: {accuracy:.4f} | Time: {epoch_time:.2f}s")
        else:
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")
    
    # Final evaluation
    print("\nFinal evaluation...")
    key, eval_key = jax.random.split(key)
    final_accuracy, final_loss = evaluate_model(state.model, test_iterator, eval_key)
    
    print(f"Final Test Accuracy: {final_accuracy:.4f}")
    print(f"Final Test Loss: {final_loss:.4f}")
    
    # Plot training curves
    plot_training_curves(losses, accuracies, "NoProp CIFAR-10 Training")
    
    return {
        "model": state.model,
        "final_accuracy": final_accuracy,
        "final_loss": final_loss,
        "losses": losses,
        "accuracies": accuracies,
        "dataset": "cifar10"
    }


def demonstrate_diffusion_process(
    dataset: str = "mnist",
    num_samples: int = 3,
    seed: int = 42
) -> None:
    """
    Demonstrate the diffusion process by visualizing label corruption.
    
    Args:
        dataset: Dataset to use ("mnist", "cifar10", "cifar100")
        num_samples: Number of samples to show
        seed: Random seed
    """
    print(f"Demonstrating diffusion process on {dataset.upper()}...")
    
    key = jax.random.PRNGKey(seed)
    
    # Load data
    if dataset == "mnist":
        _, test_iterator = load_mnist_data(batch_size=32)
    elif dataset == "cifar10":
        _, test_iterator = load_cifar10_data(batch_size=32)
    elif dataset == "cifar100":
        _, test_iterator = load_cifar100_data(batch_size=32)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    dataset_info = get_dataset_info(dataset)
    
    # Get a batch of data
    for x, y in test_iterator():
        break
    
    # Take first few samples
    x_samples = x[:num_samples]
    y_samples = y[:num_samples]
    
    # Convert labels to one-hot
    clean_labels = jax.nn.one_hot(y_samples, dataset_info["num_classes"])
    
    # Create noise schedule
    T = 10
    alpha_schedule = create_noise_schedule(T, "linear")
    
    # Create noisy sequence
    from .training import create_noisy_sequence
    key, subkey = jax.random.split(key)
    noisy_sequence = create_noisy_sequence(clean_labels, alpha_schedule, subkey)
    
    # Visualize
    visualize_diffusion_process(
        clean_labels, 
        noisy_sequence, 
        alpha_schedule,
        class_names=dataset_info["class_names"]
    )


def demonstrate_inference_process(
    trained_model,
    dataset: str = "mnist",
    seed: int = 42
) -> None:
    """
    Demonstrate the inference (reverse diffusion) process.
    
    Args:
        trained_model: Trained NoProp model
        dataset: Dataset name
        seed: Random seed
    """
    print(f"Demonstrating inference process on {dataset.upper()}...")
    
    key = jax.random.PRNGKey(seed)
    
    # Load test data
    if dataset == "mnist":
        _, test_iterator = load_mnist_data(batch_size=32)
    elif dataset == "cifar10":
        _, test_iterator = load_cifar10_data(batch_size=32)
    elif dataset == "cifar100":
        _, test_iterator = load_cifar100_data(batch_size=32)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    dataset_info = get_dataset_info(dataset)
    
    # Get a batch of test data
    for x, y in test_iterator():
        break
    
    # Take first sample
    x_sample = x[:1]
    y_sample = y[:1]
    
    # Perform inference with intermediate states
    key, subkey = jax.random.split(key)
    predictions, intermediate_states = inference_with_intermediate(trained_model, x_sample, subkey)
    
    # Visualize the inference process
    from .utils import visualize_inference_process
    visualize_inference_process(
        predictions,
        intermediate_states,
        int(y_sample[0]),
        class_names=dataset_info["class_names"]
    )


def benchmark_performance(
    dataset: str = "mnist",
    batch_sizes: list = [32, 64, 128, 256],
    T_values: list = [5, 10, 20],
    seed: int = 42
) -> Dict[str, Any]:
    """
    Benchmark performance for different configurations.
    
    Args:
        dataset: Dataset to benchmark on
        batch_sizes: List of batch sizes to test
        T_values: List of diffusion step counts to test
        seed: Random seed
        
    Returns:
        Benchmark results
    """
    print(f"Benchmarking performance on {dataset.upper()}...")
    
    key = jax.random.PRNGKey(seed)
    dataset_info = get_dataset_info(dataset)
    
    results = {
        "batch_sizes": [],
        "T_values": [],
        "train_times": [],
        "inference_times": [],
        "memory_usage": []
    }
    
    for T in T_values:
        for batch_size in batch_sizes:
            print(f"\nTesting T={T}, batch_size={batch_size}")
            
            # Initialize model
            key, model_key = jax.random.split(key)
            model = init_noprop_model(
                key=model_key,
                T=T,
                embed_dim=dataset_info["num_classes"],
                input_channels=dataset_info["input_channels"]
            )
            
            # Create dummy data
            dummy_x = jnp.ones((batch_size, dataset_info["input_channels"], *dataset_info["input_size"]))
            dummy_y = jnp.zeros(batch_size, dtype=jnp.int32)
            
            # Benchmark training step
            key, train_key = jax.random.split(key)
            optimizer = optax.adamw(learning_rate=1e-3)
            state = create_train_state(model)
            
            # Warmup
            for _ in range(3):
                key, subkey = jax.random.split(key)
                state, _ = train_step(state, dummy_x, dummy_y, subkey, optimizer)
            
            # Time training
            start_time = time.time()
            for _ in range(10):
                key, subkey = jax.random.split(key)
                state, _ = train_step(state, dummy_x, dummy_y, subkey, optimizer)
            train_time = (time.time() - start_time) / 10
            
            # Benchmark inference
            key, inf_key = jax.random.split(key)
            
            # Warmup
            for _ in range(3):
                key, subkey = jax.random.split(key)
                _ = inference_step(state.model, dummy_x, subkey)
            
            # Time inference
            start_time = time.time()
            for _ in range(10):
                key, subkey = jax.random.split(key)
                _ = inference_step(state.model, dummy_x, subkey)
            inference_time = (time.time() - start_time) / 10
            
            # Store results
            results["T_values"].append(T)
            results["batch_sizes"].append(batch_size)
            results["train_times"].append(train_time)
            results["inference_times"].append(inference_time)
            
            print(f"  Train time: {train_time:.4f}s")
            print(f"  Inference time: {inference_time:.4f}s")
    
    return results
