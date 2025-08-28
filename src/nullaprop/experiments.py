"""
Experiments and benchmarks for NoProp JAX implementation.
"""

import jax
import jax.numpy as jnp
import optax
import time
from tqdm import tqdm
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt

from .models import init_noprop_model # This is the new init_noprop_model
from .training import create_train_state, train_step, compute_loss_aligned
from .inference import inference_ct_euler, inference_ct_heun # New inference functions
# inference_with_intermediate might need an update or replacement if its logic was tied to old model
from .utils import (
    load_data,
    get_dataset_info, print_model_summary, evaluate_model, # evaluate_model uses old inference_step
    plot_training_curves, # visualize_diffusion_process might be less relevant for CT
    # create_noise_schedule # Obsolete, model has NoiseSchedule class
    initialize_with_prototypes_jax # Added
)


def run_experiment(
    dataset: str = 'mnist',
    epochs: int = 200,
    batch_size: int = 2048,
    learning_rate: float = 1e-3,
    T: int = 40,
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
    
    # Load data
    print(f"Loading {dataset.upper()} dataset...")
    train_iterator, test_iterator = load_data(dataset, batch_size=batch_size)
    dataset_info = get_dataset_info(dataset)
    
    # Initialize model (using new init_noprop_model signature)
    print("Initializing NoProp model (CT version)...")
    model = init_noprop_model(
        key=model_key,
        num_classes=dataset_info["num_classes"],
        input_channels=dataset_info["input_channels"],
        embed_dim=256,
    )
    
    # Print model summary (needs update for new model structure)
    # print_model_summary(model, dataset_info["input_size"]) # Old summary fn
    print(f"Model initialized: embed_dim={model.embed_dim}, num_classes={model.num_classes}")

    # Create training state
    optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=1e-3)
    # create_train_state now requires a key
    key, state_key = jax.random.split(key)
    state = create_train_state(model, optimizer, key=state_key)
    
    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    losses = []
    accuracies = []
    
    for epoch in (pbar := tqdm(range(1, epochs + 1))):
        epoch_loss = 0.0
        num_batches = 0
        
        # Training
        for x, y in train_iterator:
            # train_step now takes optimizer directly, key is in state
            state, loss = train_step(state, x, y, optimizer, eta=1.0) # Assuming eta=1.0
            epoch_loss += loss
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        losses.append(float(avg_loss))
        
        # Evaluation every 5 epochs
        if epoch % 5 == 0 or epoch == epochs:
            eval_key, state_key_new = jax.random.split(state.key)
            state = state._replace(key=state_key_new) # Update state's key

            # evaluate_model needs to be updated or use a local evaluation loop
            # For now, let's implement a local eval loop using new inference
            test_correct = 0
            test_total = 0
            test_loss_sum = 0
            test_batches = 0
            for x_test, y_test in test_iterator:
                key_inf, eval_key = jax.random.split(eval_key)
                preds = inference_ct_heun(state.model, x_test, key_inf, T_steps=T) # Example T_steps
                test_correct += jnp.sum(preds == y_test)
                test_total += len(y_test)
                
                key_loss, eval_key = jax.random.split(eval_key)
                loss_val = compute_loss_aligned(state.model, x_test, y_test, key_loss, eta=1.0)
                test_loss_sum += loss_val
                test_batches +=1

            accuracy = test_correct / test_total if test_total > 0 else 0.0
            eval_loss = test_loss_sum / test_batches if test_batches > 0 else 0.0
            accuracies.append(float(accuracy))

        pbar.set_postfix(loss=f'{avg_loss:.4f}')
    
    # Final evaluation
    print("\nFinal evaluation...")
    final_eval_key, _ = jax.random.split(state.key) # Use final state's key
    # Full evaluation using local loop
    test_correct_final = 0
    test_total_final = 0
    test_loss_sum_final = 0
    test_batches_final = 0
    for x_test, y_test in test_iterator:
        key_inf, final_eval_key = jax.random.split(final_eval_key)
        preds = inference_ct_euler(state.model, x_test, key_inf, T_steps=100) # More steps for final
        test_correct_final += jnp.sum(preds == y_test)
        test_total_final += len(y_test)
        
        key_loss, final_eval_key = jax.random.split(final_eval_key)
        loss_val = compute_loss_aligned(state.model, x_test, y_test, key_loss, eta=1.0)
        test_loss_sum_final += loss_val
        test_batches_final +=1
        
    final_accuracy = test_correct_final / test_total_final if test_total_final > 0 else 0.0
    final_loss = test_loss_sum_final / test_batches_final if test_batches_final > 0 else 0.0
    
    return {
        "model": state.model,
        "final_accuracy": final_accuracy,
        "final_loss": final_loss,
        "losses": losses,
        "accuracies": accuracies,
        "dataset": "mnist"
    }

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
        "inference_times_euler": [],
        "inference_times_heun": [],
    }
    
    for batch_size in batch_sizes:
        # Initialize model (CT version)
        key, model_key = jax.random.split(key)
        model = init_noprop_model(
            key=model_key,
            num_classes=dataset_info["num_classes"],
            embed_dim=256,
            input_channels=dataset_info["input_channels"],
        )
        
        # Create dummy data
        dummy_x = jnp.ones(
            (batch_size, dataset_info["input_channels"], *dataset_info["input_size"]), dtype=jnp.float32
        )
        dummy_y = jnp.zeros(batch_size, dtype=jnp.int32)
        
        # Benchmark training step
        key, state_key = jax.random.split(key)
        optimizer = optax.adamw(learning_rate=1e-3, weight_decay=1e-3)
        state = create_train_state(model, optimizer, key=state_key)
        
        # Warmup
        train_step(state, dummy_x, dummy_y, optimizer, eta=1.0)
        
        # Time training
        start_time = time.time()
        for _ in range(10): # Number of steps to average over
            state, loss = train_step(state, dummy_x, dummy_y, optimizer, eta=1.0)
        loss.block_until_ready()
        train_time = (time.time() - start_time) / 10
        
        print(f"\nbatch_size={batch_size}")
        print(f"Train time: {train_time:.4f}s")
        for T in T_values:
            print(f"\nTesting T={T}")
            # Benchmark inference
            
            # Warmup
            key_warmup, _ = jax.random.split(state.key) # Use key from state
            key_inf_loop, key_warmup = jax.random.split(key_warmup)
            inference_ct_euler(state.model, dummy_x, key_inf_loop, T_steps=T)
            
            # Time inference
            key_time_inf, _ = jax.random.split(key_warmup)
            start_time = time.time()
            for _ in range(10): # Number of inference calls to average
                key_inf_loop, key_time_inf = jax.random.split(key_time_inf)
                pred = inference_ct_euler(state.model, dummy_x, key_inf_loop, T_steps=T)
            pred.block_until_ready()
            inference_time_euler = (time.time() - start_time) / 10

            key_warmup, _ = jax.random.split(key_time_inf)
            key_inf_loop, key_warmup = jax.random.split(key_warmup)
            inference_ct_heun(state.model, dummy_x, key_inf_loop, T_steps=T)
            
            # Time inference
            key_time_inf, _ = jax.random.split(key_warmup)
            start_time = time.time()
            for _ in range(10): # Number of inference calls to average
                key_inf_loop, key_time_inf = jax.random.split(key_time_inf)
                pred = inference_ct_heun(state.model, dummy_x, key_inf_loop, T_steps=T)
            pred.block_until_ready()
            inference_time_heun = (time.time() - start_time) / 10
            
            # Store results
            results["T_values"].append(T) # Store T_steps used for inference
            results["batch_sizes"].append(batch_size)
            results["train_times"].append(train_time)
            results["inference_times_euler"].append(inference_time_euler)
            results["inference_times_heun"].append(inference_time_heun)
            
            print(f"  Inference time Euler: {inference_time_euler:.4f}s")
            print(f"  Inference time Heun: {inference_time_heun:.4f}s")
    
    return results
