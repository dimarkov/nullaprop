"""
Training utilities for NoProp implementation.
"""

import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from typing import Tuple, NamedTuple, Any
from .models import NoPropModel, apply_mlp


class TrainState(NamedTuple):
    """Training state containing model parameters and optimizer state."""
    model: NoPropModel
    optimizer_state: Any
    step: int


def diffusion_scan_step(carry: Tuple[jnp.ndarray, jax.random.PRNGKey], 
                       inputs: Tuple[float, jnp.ndarray]) -> Tuple[Tuple[jnp.ndarray, jax.random.PRNGKey], jnp.ndarray]:
    """
    Single diffusion step for scan operation.
    
    Args:
        carry: (z_prev, key) - previous latent state and random key
        inputs: (alpha_t, mlp_params_t) - noise schedule value and MLP parameters
        
    Returns:
        Updated carry and current noisy state
    """
    z_prev, key = carry
    alpha_t, _ = inputs  # We don't use mlp_params here, just for scan structure
    
    # Generate noise
    key, subkey = jax.random.split(key)
    noise = jax.random.normal(subkey, z_prev.shape)
    
    # Forward diffusion: z_t = √α_t * z_{t-1} + √(1-α_t) * ε
    z_t = jnp.sqrt(alpha_t) * z_prev + jnp.sqrt(1 - alpha_t) * noise
    
    return (z_t, key), z_t


def create_noisy_sequence(clean_labels: jnp.ndarray, 
                         alpha_schedule: jnp.ndarray,
                         key: jax.random.PRNGKey) -> jnp.ndarray:
    """
    Create sequence of noisy labels using scan for memory efficiency.
    
    Args:
        clean_labels: One-hot encoded labels [batch_size, embed_dim]
        alpha_schedule: Noise schedule [T]
        key: Random key for noise generation
        
    Returns:
        Noisy label sequence [T, batch_size, embed_dim]
    """
    # Initialize carry
    carry_init = (clean_labels, key)
    
    # Create scan inputs (alpha values and dummy mlp_params)
    T = len(alpha_schedule)
    dummy_mlp_params = jnp.zeros((T, 1))  # Not used in diffusion, just for scan structure
    scan_inputs = (alpha_schedule, dummy_mlp_params)
    
    # Run scan
    _, z_sequence = jax.lax.scan(diffusion_scan_step, carry_init, scan_inputs)
    
    return z_sequence


@eqx.filter_jit
def compute_loss(model: NoPropModel, 
                x: jnp.ndarray, 
                y: jnp.ndarray,
                key: jax.random.PRNGKey,
                eta: float = 0.1) -> jnp.ndarray:
    """
    Compute NoProp training loss following Equation 8 from the paper.
    
    L_NoProp = E_q(z_T|y)[-log p̂_θ_out(y|z_T)] + D_KL(q(z_0|y)||p(z_0)) + 
               (T/2)η E_t~U{1,T}[(SNR(t)-SNR(t-1))||û_θ_t(z_{t-1},x) - u_y||²]
    
    Args:
        model: NoProp model
        x: Input images [batch_size, channels, height, width]
        y: Labels [batch_size]
        key: Random key for noise generation
        eta: Hyperparameter η from the paper
        
    Returns:
        Total loss scalar
    """
    batch_size = x.shape[0]
    T = model.T
    
    # Convert labels to embeddings
    u_y = model.embed_matrix[y]  # [batch_size, embed_dim]
    embed_dim = u_y.shape[1]  # Get embed_dim from actual shape, not model attribute
    
    # Extract image features
    x_features = model.extract_features(x)  # [batch_size, feature_dim]
    
    # Compute cumulative alphas (ᾱ_t)
    alpha_bar = jnp.cumprod(model.alpha_schedule)  # [T]
    
    # === 1. Classification Loss: E_q(z_T|y)[-log p̂_θ_out(y|z_T)] ===
    key, subkey = jax.random.split(key)
    # Sample z_T ~ q(z_T|y) = N(√ᾱ_T u_y, 1-ᾱ_T)
    noise = jax.random.normal(subkey, (batch_size, embed_dim))
    z_T = jnp.sqrt(alpha_bar[T-1]) * u_y + jnp.sqrt(1 - alpha_bar[T-1]) * noise
    
    # Compute classification probabilities
    # Use vmap to apply classifier to each sample in the batch
    logits = jax.vmap(model.classifier)(z_T)  # [batch_size, num_classes]
    classification_loss = -jnp.mean(jax.nn.log_softmax(logits)[jnp.arange(batch_size), y])
    
    # === 2. KL Divergence: D_KL(q(z_0|y)||p(z_0)) ===
    # q(z_0|y) = N(√ᾱ_0 u_y, 1-ᾱ_0), p(z_0) = N(0, I)
    # For standard diffusion: α_0 = 1.0, so ᾱ_0 = 1.0
    # This means q(z_0|y) = N(u_y, 0) = δ(u_y) and KL = ||u_y||²/2
    kl_loss = 0.5 * jnp.mean(jnp.sum(u_y**2, axis=1))
    
    # === 3. Denoising Loss: (T/2)η E_t[(SNR(t)-SNR(t-1))||û_θ_t(z_{t-1},x) - u_y||²] ===
    def compute_timestep_loss(t: int, key: jax.random.PRNGKey) -> jnp.ndarray:
        """Compute loss for a specific timestep t."""
        # Sample z_{t-1} ~ q(z_{t-1}|y) = N(√ᾱ_{t-1} u_y, 1-ᾱ_{t-1})
        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, (batch_size, embed_dim))
        
        # Use JAX conditional for traced values
        def get_z_prev_t0():
            # For t=0, z_{-1} doesn't exist, so we use z_0 = u_y (clean labels)
            return u_y
        
        def get_z_prev_other():
            alpha_bar_prev = alpha_bar[jnp.maximum(t-1, 0)]  # Prevent negative indexing
            return jnp.sqrt(alpha_bar_prev) * u_y + jnp.sqrt(1 - alpha_bar_prev) * noise
        
        z_prev = jax.lax.cond(t == 0, get_z_prev_t0, get_z_prev_other)
        
        # Get model prediction û_θ_t(z_{t-1}, x)
        u_hat = model.apply_single_mlp(t, x_features, z_prev)
        
        # Compute SNR terms using JAX conditional
        def get_snr_diff_t0():
            return alpha_bar[0] / (1 - alpha_bar[0])  # SNR(0) - SNR(-1), with SNR(-1) = 0
        
        def get_snr_diff_other():
            snr_t = alpha_bar[t] / (1 - alpha_bar[t])
            snr_prev = alpha_bar[jnp.maximum(t-1, 0)] / (1 - alpha_bar[jnp.maximum(t-1, 0)])
            return snr_t - snr_prev
        
        snr_diff = jax.lax.cond(t == 0, get_snr_diff_t0, get_snr_diff_other)
        
        # MSE loss weighted by SNR difference
        mse_loss = jnp.mean(jnp.sum((u_hat - u_y)**2, axis=1))
        
        return snr_diff * mse_loss
    
    # Sample a random timestep for unbiased estimation
    key, subkey = jax.random.split(key)
    t_sample = jax.random.randint(subkey, (), 0, T)
    
    # Compute denoising loss for sampled timestep
    key, subkey = jax.random.split(key)
    denoising_loss = compute_timestep_loss(t_sample, subkey)
    
    # Scale by (T/2) * η as in the paper
    denoising_loss = (T / 2.0) * eta * denoising_loss
    
    # === Total Loss ===
    total_loss = classification_loss + kl_loss + denoising_loss
    
    return total_loss


def train_step(state: TrainState,
               x: jnp.ndarray,
               y: jnp.ndarray,
               key: jax.random.PRNGKey,
               optimizer: optax.GradientTransformation) -> Tuple[TrainState, jnp.ndarray]:
    """
    Single training step.
    
    Args:
        state: Current training state
        x: Input batch [batch_size, channels, height, width]
        y: Label batch [batch_size]
        key: Random key
        optimizer: Optax optimizer
        
    Returns:
        Updated training state and loss value
    """
    # Use Equinox filter_grad to handle mixed static/trainable parameters
    loss, grads = eqx.filter_value_and_grad(compute_loss)(state.model, x, y, key)
    
    # Update parameters
    updates, new_optimizer_state = optimizer.update(grads, state.optimizer_state, state.model)
    new_model = eqx.apply_updates(state.model, updates)
    
    # Create new state
    new_state = TrainState(
        model=new_model,
        optimizer_state=new_optimizer_state,
        step=state.step + 1
    )
    
    return new_state, loss


def create_train_state(model: NoPropModel,
                      learning_rate: float = 1e-3,
                      weight_decay: float = 1e-4) -> TrainState:
    """
    Create initial training state.
    
    Args:
        model: NoProp model
        learning_rate: Learning rate
        weight_decay: Weight decay coefficient
        
    Returns:
        Initial training state
    """
    # Create optimizer
    optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    
    # Initialize optimizer state
    optimizer_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    
    return TrainState(
        model=model,
        optimizer_state=optimizer_state,
        step=0
    )


def train_epoch(state: TrainState,
               data_loader: Any,
               optimizer: optax.GradientTransformation,
               key: jax.random.PRNGKey) -> Tuple[TrainState, float]:
    """
    Train for one epoch.
    
    Args:
        state: Current training state
        data_loader: Data loader yielding (x, y) batches
        optimizer: Optax optimizer
        key: Random key
        
    Returns:
        Updated training state and average loss
    """
    epoch_loss = 0.0
    num_batches = 0
    
    for batch_idx, (x, y) in enumerate(data_loader):
        # Split key for this batch
        key, subkey = jax.random.split(key)
        
        # Convert numpy arrays to JAX if needed
        if hasattr(x, 'numpy'):
            x = jnp.array(x.numpy())
            y = jnp.array(y.numpy())
        else:
            x = jnp.array(x)
            y = jnp.array(y)
        
        # Ensure correct shape for CNN (add channel dimension if needed)
        if len(x.shape) == 3:  # [batch, height, width]
            x = x[:, None, :, :]  # [batch, 1, height, width]
        
        # Training step
        state, loss = train_step(state, x, y, subkey, optimizer)
        
        epoch_loss += loss
        num_batches += 1
    
    avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
    
    return state, avg_loss


@eqx.filter_jit  
def compute_accuracy(model: NoPropModel, x: jnp.ndarray, y: jnp.ndarray, key: jax.random.PRNGKey) -> float:
    """
    Compute classification accuracy.
    
    Args:
        model: NoProp model
        x: Input images
        y: True labels
        key: Random key for inference
        
    Returns:
        Accuracy as float
    """
    from .inference import inference_step
    
    # Get predictions
    predictions = inference_step(model, x, key)
    
    # Compute accuracy
    correct = jnp.sum(predictions == y)
    total = len(y)
    
    return correct / total
