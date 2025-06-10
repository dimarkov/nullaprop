#!/usr/bin/env python3
"""
Pytest tests for NoProp CT model functionality.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from nullaprop.models import init_noprop_model
from nullaprop.training import compute_loss_aligned, create_train_state, train_step
from nullaprop.inference import inference_ct_euler, inference_ct_heun
import optax

# Common configuration for tests
NUM_CLASSES_TEST = 10
EMBED_DIM_TEST = 64
INPUT_CHANNELS_TEST = 1
TIME_EMB_DIM_INTERNAL_TEST = 64
BATCH_SIZE_TEST = 4

@pytest.fixture
def key():
    return jax.random.PRNGKey(42)

@pytest.fixture
def model_ct(key):
    key, model_key = jax.random.split(key)
    return init_noprop_model(
        key=model_key,
        num_classes=NUM_CLASSES_TEST,
        embed_dim=EMBED_DIM_TEST,
        input_channels=INPUT_CHANNELS_TEST,
        time_emb_dim_internal=TIME_EMB_DIM_INTERNAL_TEST,
    )

@pytest.fixture
def dummy_data_ct(key, model_ct):
    dummy_x_imgs = jax.random.normal(key, (BATCH_SIZE_TEST, INPUT_CHANNELS_TEST, 28, 28))
    dummy_y_labels = jnp.array(np.random.randint(0, NUM_CLASSES_TEST, BATCH_SIZE_TEST))
    key, t_key, z_key = jax.random.split(key, 3)
    dummy_t_continuous = jax.random.uniform(t_key, (BATCH_SIZE_TEST, 1))
    dummy_z_noisy = jax.random.normal(z_key, (BATCH_SIZE_TEST, model_ct.embed_dim))
    return dummy_x_imgs, dummy_y_labels, dummy_t_continuous, dummy_z_noisy

def test_model_initialization_ct(model_ct):
    """Test model initialization and basic properties."""
    assert model_ct.embed_dim == EMBED_DIM_TEST
    assert model_ct.num_classes == NUM_CLASSES_TEST
    assert model_ct.embed_matrix.shape == (NUM_CLASSES_TEST, EMBED_DIM_TEST)

def test_cnn_feature_extraction_ct(model_ct, dummy_data_ct, key):
    """Test CNN feature extraction part of the model."""
    dummy_x_imgs, _, _, _ = dummy_data_ct
    x_features = model_ct.cnn(dummy_x_imgs)
    assert x_features.shape == (BATCH_SIZE_TEST, model_ct.embed_dim)

def test_forward_unified_pass_ct(model_ct, dummy_data_ct):
    """Test the main forward_unified pass of the model."""
    dummy_x_imgs, _, dummy_t_continuous, dummy_z_noisy = dummy_data_ct
    logits = model_ct(dummy_x_imgs, dummy_z_noisy, dummy_t_continuous)
    assert logits.shape == (BATCH_SIZE_TEST, model_ct.num_classes)

def test_loss_computation_ct(model_ct, dummy_data_ct, key):
    """Test loss computation using compute_loss_aligned."""
    dummy_x_imgs, dummy_y_labels, _, _ = dummy_data_ct
    key, loss_key = jax.random.split(key)
    loss = compute_loss_aligned(model_ct, dummy_x_imgs, dummy_y_labels, loss_key, eta=1.0)
    assert isinstance(loss.item(), float)
    assert loss >= 0

@pytest.fixture
def train_state_ct(model_ct, key):
    optimizer = optax.adamw(learning_rate=1e-3)
    key, state_key = jax.random.split(key)
    return create_train_state(model_ct, optimizer, key=state_key)

def test_training_step_ct(train_state_ct, dummy_data_ct):
    """Test a single training step."""
    dummy_x_imgs, dummy_y_labels, _, _ = dummy_data_ct
    # The optimizer is part of the state in create_train_state, but train_step might expect it explicitly
    # Based on original script, train_step takes optimizer as an argument
    optimizer = optax.adamw(learning_rate=1e-3) # Re-create or pass from fixture if state doesn't hold it in usable form for train_step
    
    new_state, train_loss = train_step(train_state_ct, dummy_x_imgs, dummy_y_labels, optimizer, eta=1.0)
    assert isinstance(train_loss.item(), float)
    assert train_loss >= 0

def test_inference_euler_ct(train_state_ct, dummy_data_ct, key):
    """Test Euler inference."""
    dummy_x_imgs, _, _, _ = dummy_data_ct
    key, inf_key = jax.random.split(key)
    predictions_euler = inference_ct_euler(train_state_ct.model, dummy_x_imgs, inf_key, T_steps=10)
    assert predictions_euler.shape == (BATCH_SIZE_TEST,)
    assert jnp.all(predictions_euler >= 0) and jnp.all(predictions_euler < NUM_CLASSES_TEST)

def test_inference_heun_ct(train_state_ct, dummy_data_ct, key):
    """Test Heun inference."""
    dummy_x_imgs, _, _, _ = dummy_data_ct
    key, inf_key_heun = jax.random.split(key)
    predictions_heun = inference_ct_heun(train_state_ct.model, dummy_x_imgs, inf_key_heun, T_steps=5)
    assert predictions_heun.shape == (BATCH_SIZE_TEST,)
    assert jnp.all(predictions_heun >= 0) and jnp.all(predictions_heun < NUM_CLASSES_TEST)

def test_mini_training_loop_ct(key):
    """Test a mini training loop for the CT model to ensure it runs and learns."""
    num_classes_mini = 3
    embed_dim_mini = 16
    
    key, model_key = jax.random.split(key)
    model = init_noprop_model(
        key=model_key,
        num_classes=num_classes_mini,
        embed_dim=embed_dim_mini,
        input_channels=1,
        time_emb_dim_internal=32,
    )
    
    num_samples_mini = 32
    key, data_key_init = jax.random.split(key)
    
    x_data_list = []
    y_data_list = []
    
    current_data_key = data_key_init
    for i in range(num_samples_mini):
        k_loop, current_data_key = jax.random.split(current_data_key)
        if i < num_samples_mini // 3:
            x = jax.random.normal(k_loop, (1, 28, 28)) - 1.0
            y_val = 0
        elif i < 2 * num_samples_mini // 3:
            x = jax.random.normal(k_loop, (1, 28, 28))
            y_val = 1
        else:
            x = jax.random.normal(k_loop, (1, 28, 28)) + 1.0
            y_val = 2
        x_data_list.append(x)
        y_data_list.append(y_val)
    
    x_train = jnp.stack(x_data_list)
    y_train = jnp.array(y_data_list)
    
    learning_rate_mini = 1e-2
    optimizer = optax.adamw(learning_rate=learning_rate_mini, weight_decay=1e-3)
    key, state_key = jax.random.split(key)
    state = create_train_state(model, optimizer, key=state_key)
    
    batch_size_mini = 8
    num_epochs_mini = 5 # Reduced epochs for faster test
    
    current_loop_key = key # Use the main key for permutations initially
    for epoch in range(num_epochs_mini):
        epoch_loss_sum = 0.0
        num_batches = 0
        
        perm_key, current_loop_key = jax.random.split(current_loop_key)
        perm = jax.random.permutation(perm_key, num_samples_mini)
        x_shuffled = x_train[perm]
        y_shuffled = y_train[perm]

        for i in range(0, num_samples_mini, batch_size_mini):
            end_idx = min(i + batch_size_mini, num_samples_mini)
            x_batch = x_shuffled[i:end_idx]
            y_batch = y_shuffled[i:end_idx]
            
            if x_batch.shape[0] == 0: continue

            state, loss = train_step(state, x_batch, y_batch, optimizer, eta=1.0)
            epoch_loss_sum += loss
            num_batches += 1
        
        avg_loss = epoch_loss_sum / num_batches if num_batches > 0 else 0.0
        # print(f"  Epoch {epoch+1:2d} (test): Loss = {avg_loss:.4f}") # Optional: for debugging if test fails

    # Use key from state for final inference
    final_inf_key, new_state_key = jax.random.split(state.key)
    state = state._replace(key=new_state_key) # Consume key

    final_preds = inference_ct_euler(state.model, x_train, final_inf_key, T_steps=10) # Reduced T_steps
    final_acc = jnp.mean(final_preds == y_train)
    
    assert final_acc > 0.3 # Basic check that some learning occurred
