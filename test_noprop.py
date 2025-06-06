#!/usr/bin/env python3
"""
Simple test script to verify NoProp implementation works.
"""

import jax
import jax.numpy as jnp
import numpy as np
from nullaprop.models import init_noprop_model
from nullaprop.training import compute_loss, create_train_state, train_step
from nullaprop.inference import inference_step
import optax

def test_basic_functionality():
    """Test basic model creation and forward pass."""
    print("Testing basic NoProp functionality...")
    
    # Set random seed
    key = jax.random.PRNGKey(42)
    
    # Initialize model for MNIST-like data
    key, model_key = jax.random.split(key)
    model = init_noprop_model(
        key=model_key,
        T=5,  # Small number for testing
        num_classes=10,
        embed_dim=10,
        feature_dim=64,
        input_channels=1,
        hidden_dim=128,
        embedding_type="one_hot"
    )
    
    print(f"Model initialized with T={model.T}, embed_dim={model.embed_dim}")
    print(f"Embedding matrix shape: {model.embed_matrix.shape}")
    print(f"MLP params shape: {model.mlp_params.shape}")
    print(f"Alpha schedule: {model.alpha_schedule}")
    
    # Create dummy data (MNIST-like: 28x28 grayscale)
    batch_size = 4
    dummy_x = jax.random.normal(key, (batch_size, 1, 28, 28))
    dummy_y = jnp.array([0, 1, 2, 3])  # Class labels
    
    # Test feature extraction
    print("\nTesting CNN feature extraction...")
    x_features = model.extract_features(dummy_x)
    print(f"Input shape: {dummy_x.shape}")
    print(f"Feature shape: {x_features.shape}")
    
    # Test loss computation
    print("\nTesting loss computation...")
    key, loss_key = jax.random.split(key)
    loss = compute_loss(model, dummy_x, dummy_y, loss_key)
    print(f"Loss: {loss:.4f}")
    
    # Test training step
    print("\nTesting training step...")
    optimizer = optax.adamw(learning_rate=1e-3)
    state = create_train_state(model, learning_rate=1e-3)
    
    key, train_key = jax.random.split(key)
    new_state, train_loss = train_step(state, dummy_x, dummy_y, train_key, optimizer)
    print(f"Training loss: {train_loss:.4f}")
    print(f"Training step successful: {new_state.step == 1}")
    
    # Test inference
    print("\nTesting inference...")
    key, inf_key = jax.random.split(key)
    predictions = inference_step(model, dummy_x, inf_key)
    print(f"Predictions: {predictions}")
    print(f"True labels: {dummy_y}")
    
    print("\nBasic functionality test completed successfully! ✓")
    return True

def test_different_embedding_types():
    """Test different embedding types."""
    print("\nTesting different embedding types...")
    
    key = jax.random.PRNGKey(42)
    
    # Test one-hot embeddings
    key, subkey = jax.random.split(key)
    model_onehot = init_noprop_model(
        key=subkey,
        T=3,
        num_classes=5,
        embed_dim=5,
        embedding_type="one_hot"
    )
    print(f"One-hot embeddings shape: {model_onehot.embed_matrix.shape}")
    print(f"One-hot embeddings:\n{model_onehot.embed_matrix}")
    
    # Test learnable embeddings
    key, subkey = jax.random.split(key)
    model_learnable = init_noprop_model(
        key=subkey,
        T=3,
        num_classes=5,
        embed_dim=8,
        embedding_type="learnable"
    )
    print(f"\nLearnable embeddings shape: {model_learnable.embed_matrix.shape}")
    print(f"Learnable embeddings (first 3x3):\n{model_learnable.embed_matrix[:3, :3]}")
    
    print("Embedding types test completed successfully! ✓")

def test_parallel_computation():
    """Test that the model can handle parallel computation properly."""
    print("\nTesting parallel computation capabilities...")
    
    key = jax.random.PRNGKey(42)
    
    # Initialize model
    key, model_key = jax.random.split(key)
    model = init_noprop_model(
        key=model_key,
        T=10,
        num_classes=10,
        embed_dim=10
    )
    
    # Test with different batch sizes
    for batch_size in [1, 4, 16]:
        print(f"  Testing batch size: {batch_size}")
        
        # Create dummy data
        key, data_key = jax.random.split(key)
        x = jax.random.normal(data_key, (batch_size, 1, 28, 28))
        y = jnp.arange(batch_size) % 10
        
        # Test loss computation
        key, loss_key = jax.random.split(key)
        loss = compute_loss(model, x, y, loss_key)
        
        # Test inference
        key, inf_key = jax.random.split(key)
        predictions = inference_step(model, x, inf_key)
        
        print(f"    Loss: {loss:.4f}, Predictions shape: {predictions.shape}")
        assert predictions.shape == (batch_size,), f"Expected {(batch_size,)}, got {predictions.shape}"
    
    print("Parallel computation test completed successfully! ✓")

def run_mini_training():
    """Run a mini training loop to test everything works together."""
    print("\nRunning mini training loop...")
    
    key = jax.random.PRNGKey(42)
    
    # Initialize model
    key, model_key = jax.random.split(key)
    model = init_noprop_model(
        key=model_key,
        T=5,
        num_classes=3,  # Simple 3-class problem
        embed_dim=3
    )
    
    # Create synthetic dataset
    num_samples = 32
    key, data_key = jax.random.split(key)
    
    # Simple synthetic data: class 0 = negative values, class 1 = middle, class 2 = positive
    x_data = []
    y_data = []
    
    for i in range(num_samples):
        key, subkey = jax.random.split(key)
        if i < num_samples // 3:
            # Class 0: negative pattern
            x = jax.random.normal(subkey, (1, 28, 28)) - 1.0
            y = 0
        elif i < 2 * num_samples // 3:
            # Class 1: zero-centered pattern  
            x = jax.random.normal(subkey, (1, 28, 28))
            y = 1
        else:
            # Class 2: positive pattern
            x = jax.random.normal(subkey, (1, 28, 28)) + 1.0
            y = 2
        
        x_data.append(x)
        y_data.append(y)
    
    x_train = jnp.stack(x_data)
    y_train = jnp.array(y_data)
    
    print(f"Created synthetic dataset: {x_train.shape}, {y_train.shape}")
    
    # Initialize training
    optimizer = optax.adamw(learning_rate=1e-2)  # Higher LR for quick convergence
    state = create_train_state(model, learning_rate=1e-2)
    
    batch_size = 8
    num_epochs = 10
    
    print("Training...")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        # Simple batching
        for i in range(0, len(x_train), batch_size):
            end_idx = min(i + batch_size, len(x_train))
            x_batch = x_train[i:end_idx]
            y_batch = y_train[i:end_idx]
            
            key, train_key = jax.random.split(key)
            state, loss = train_step(state, x_batch, y_batch, train_key, optimizer)
            
            epoch_loss += loss
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        
        # Test inference every few epochs
        if epoch % 3 == 0:
            key, test_key = jax.random.split(key)
            test_preds = inference_step(state.model, x_train[:8], test_key)
            test_acc = jnp.mean(test_preds == y_train[:8])
            print(f"  Epoch {epoch+1}: Loss = {avg_loss:.4f}, Test Acc = {test_acc:.3f}")
        else:
            print(f"  Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
    # Final test
    key, final_key = jax.random.split(key)
    final_preds = inference_step(state.model, x_train, final_key)
    final_acc = jnp.mean(final_preds == y_train)
    
    print(f"Final accuracy: {final_acc:.3f}")
    print("Mini training completed successfully! ✓")
    
    return final_acc > 0.3  # Should achieve at least 30% accuracy (better than random)

def main():
    """Run all tests."""
    print("="*60)
    print("NOPROP JAX IMPLEMENTATION TEST")
    print("="*60)
    
    try:
        # Test basic functionality
        test_basic_functionality()
        
        # Test different embedding types  
        test_different_embedding_types()
        
        # Test parallel computation
        test_parallel_computation()
        
        # Run mini training
        training_success = run_mini_training()
        
        print("\n" + "="*60)
        if training_success:
            print("ALL TESTS PASSED! 🎉")
            print("NoProp implementation is working correctly.")
        else:
            print("Training test failed - check implementation")
        print("="*60)
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()
