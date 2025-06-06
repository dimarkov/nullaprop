"""
Inference utilities for NoProp implementation.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Tuple
from .models import NoPropModel, apply_mlp


def forward_diffusion_step(carry: Tuple[jnp.ndarray, jax.random.PRNGKey, jnp.ndarray, int], 
                          inputs: Tuple[float, jnp.ndarray]) -> Tuple[Tuple[jnp.ndarray, jax.random.PRNGKey, jnp.ndarray, int], jnp.ndarray]:
    """
    Single forward diffusion step following Equation 3 from the paper.
    z_t = a_t * û_θ_t(z_{t-1}, x) + b_t * z_{t-1} + √c_t * ε_t
    
    Args:
        carry: (z_prev, key, x_features, step) - previous state, random key, image features, and current step
        inputs: (alpha_t, mlp_params_t) - noise schedule value and MLP parameters for this step
        
    Returns:
        Updated carry and current state
    """
    z_prev, key, x_features, step = carry
    alpha_t, mlp_params_t = inputs
    
    # Predict label embedding using MLP
    u_hat = apply_mlp(mlp_params_t, x_features, z_prev)
    
    # Generate noise
    key, subkey = jax.random.split(key)
    noise = jax.random.normal(subkey, z_prev.shape)
    
    # Compute coefficients from Appendix A.3
    # We need to compute a_t, b_t, c_t from the cumulative alphas
    # For now, use simplified form based on the residual structure
    alpha_bar_t = jnp.cumprod(jnp.array([alpha_t]))[0]  # This step's cumulative alpha
    cond = step == 0
    alpha_bar_prev = cond * 1.0 + (1 - cond) * alpha_bar_t / alpha_t  # Previous cumulative alpha
    
    # Coefficients from paper (simplified)
    a_t = jnp.sqrt(alpha_bar_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t))
    b_t = jnp.sqrt(alpha_bar_prev * (1 - alpha_bar_t) / (1 - alpha_bar_t))  
    c_t = (1 - alpha_bar_t) * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
    
    # Apply Equation 3: z_t = a_t * û_θ_t(z_{t-1}, x) + b_t * z_{t-1} + √c_t * ε_t
    z_t = a_t * u_hat + b_t * z_prev + jnp.sqrt(jnp.maximum(c_t, 1e-6)) * noise
    
    return (z_t, key, x_features, step + 1), z_t


@eqx.filter_jit
def inference_step(model: NoPropModel, x: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.ndarray:
    """
    Perform inference using forward diffusion process as described in the paper.
    
    Starting from z_0 ~ N(0, I), apply the forward process:
    z_t = a_t * û_θ_t(z_{t-1}, x) + b_t * z_{t-1} + √c_t * ε_t
    
    Then use the classifier to predict from z_T.
    
    Args:
        model: Trained NoProp model
        x: Input images [batch_size, channels, height, width]
        key: Random key for noise generation
        
    Returns:
        Predicted class labels [batch_size]
    """
    # Extract image features
    x_features = model.extract_features(x)
    batch_size = x.shape[0]
    
    # Start from Gaussian noise z_0 ~ N(0, I)
    key, subkey = jax.random.split(key)
    z_init = jax.random.normal(subkey, (batch_size, model.embed_dim))
    
    # Initialize carry for forward diffusion
    carry_init = (z_init, key, x_features, 0)
    scan_inputs = (model.alpha_schedule, model.mlp_params)
    
    # Perform forward diffusion using scan
    (z_final, _, _, _), _ = jax.lax.scan(forward_diffusion_step, carry_init, scan_inputs)
    
    # Use classifier to get final predictions
    logits = jax.vmap(model.classifier)(z_final)
    predictions = jnp.argmax(logits, axis=-1)
    
    return predictions


@eqx.filter_jit
def inference_step_deterministic(model: NoPropModel, x: jnp.ndarray) -> jnp.ndarray:
    """
    Perform deterministic inference (no noise in reverse process).
    
    Args:
        model: Trained NoProp model
        x: Input images [batch_size, channels, height, width]
        
    Returns:
        Predicted class labels [batch_size]
    """
    # Extract image features
    x_features = model.extract_features(x)
    batch_size = x.shape[0]
    
    # Start from zero (deterministic)
    z_t = jnp.zeros((batch_size, model.embed_dim))
    
    # Reverse diffusion without noise
    for t in reversed(range(model.T)):
        alpha_t = model.alpha_schedule[t]
        mlp_params_t = model.mlp_params[t]
        
        # Predict clean label
        u_hat = apply_mlp(mlp_params_t, x_features, z_t)
        
        # Deterministic update (no noise)
        z_t = jnp.sqrt(alpha_t) * u_hat
    
    # Get final predictions
    predictions = jnp.argmax(z_t, axis=-1)
    
    return predictions


def inference_with_intermediate(model: NoPropModel, x: jnp.ndarray, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Perform inference and return intermediate states for visualization.
    
    Args:
        model: Trained NoProp model
        x: Input images [batch_size, channels, height, width]
        key: Random key for noise generation
        
    Returns:
        Tuple of (final_predictions, intermediate_states)
        - final_predictions: [batch_size]
        - intermediate_states: [T, batch_size, embed_dim]
    """
    # Extract image features
    x_features = model.extract_features(x)
    batch_size = x.shape[0]
    
    # Start from Gaussian noise z_0 ~ N(0, I)
    key, subkey = jax.random.split(key)
    z_init = jax.random.normal(subkey, (batch_size, model.embed_dim))
    
    # Initialize carry for forward diffusion
    carry_init = (z_init, key, x_features, 0)
    scan_inputs = (model.alpha_schedule, model.mlp_params)
    
    # Perform forward diffusion and collect intermediate states
    (z_final, _, _, _), z_intermediates = jax.lax.scan(forward_diffusion_step, carry_init, scan_inputs)
    
    # Use classifier to get final predictions
    logits = jax.vmap(model.classifier)(z_final)
    predictions = jnp.argmax(logits, axis=-1)
    
    return predictions, z_intermediates


@eqx.filter_jit
def batch_inference(model: NoPropModel, x: jnp.ndarray, key: jax.random.PRNGKey, batch_size: int = 128) -> jnp.ndarray:
    """
    Perform inference on large datasets by batching.
    
    Args:
        model: Trained NoProp model
        x: Input images [num_samples, channels, height, width]
        key: Random key for noise generation
        batch_size: Batch size for processing
        
    Returns:
        Predicted class labels [num_samples]
    """
    num_samples = x.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    predictions = []
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)
        
        # Get batch
        x_batch = x[start_idx:end_idx]
        
        # Split key for this batch
        key, subkey = jax.random.split(key)
        
        # Inference on batch
        batch_preds = inference_step(model, x_batch, subkey)
        predictions.append(batch_preds)
    
    return jnp.concatenate(predictions, axis=0)


def compute_prediction_confidence(model: NoPropModel, x: jnp.ndarray, key: jax.random.PRNGKey, num_samples: int = 10) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute prediction confidence using multiple stochastic forward passes.
    
    Args:
        model: Trained NoProp model
        x: Input images [batch_size, channels, height, width]
        key: Random key for noise generation
        num_samples: Number of stochastic samples
        
    Returns:
        Tuple of (mean_predictions, prediction_std)
    """
    predictions_list = []
    
    for i in range(num_samples):
        key, subkey = jax.random.split(key)
        
        # Get raw output (before argmax) for confidence estimation
        x_features = model.extract_features(x)
        batch_size = x.shape[0]
        
        # Start from random noise
        key_noise, subkey = jax.random.split(subkey)
        z_init = jax.random.normal(key_noise, (batch_size, model.embed_dim))
        
        # Forward diffusion
        carry_init = (z_init, subkey, x_features, 0)
        scan_inputs = (model.alpha_schedule, model.mlp_params)
        
        (z_final, _, _, _), _ = jax.lax.scan(forward_diffusion_step, carry_init, scan_inputs)
        
        # Get logits from classifier and apply softmax
        logits = jax.vmap(model.classifier)(z_final)
        probs = jax.nn.softmax(logits, axis=-1)
        predictions_list.append(probs)
    
    # Stack and compute statistics
    all_predictions = jnp.stack(predictions_list, axis=0)  # [num_samples, batch_size, num_classes]
    
    mean_predictions = jnp.mean(all_predictions, axis=0)
    prediction_std = jnp.std(all_predictions, axis=0)
    
    return mean_predictions, prediction_std
