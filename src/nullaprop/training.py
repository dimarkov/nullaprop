"""
Training utilities for NoProp implementation.
"""

import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from typing import Tuple, NamedTuple, Any, Optional
from .models import NoPropCT


class TrainState(NamedTuple):
    """Training state containing model parameters and optimizer state."""
    model: NoPropCT
    optimizer_state: Any
    key: jax.random.PRNGKey # Add key to train state for stateless operations


def compute_loss(model: NoPropCT, 
                 x_imgs: jnp.ndarray, 
                 y_labels: jnp.ndarray,
                 key: jax.random.PRNGKey,
                 eta: float = 1.0,
                 mask: jnp.ndarray = None, 
                 return_individual_losses: bool = False) -> jnp.ndarray:
    """
    Compute NoProp training loss, aligned with yhgon/NoProp (PyTorch version).
    L = loss_ce + loss_kl + loss_sdm
    
    Args:
        model: The new NoPropModel instance.
        x_imgs: Input images [batch_size, channels, height, width].
        y_labels: Integer class labels [batch_size, ...].
        key: JAX random key.
        eta: Hyperparameter η.
        mask: Label mask
        
    Returns:
        Total loss scalar.
    """
    batch_size = x_imgs.shape[0]
    
    # Get clean label embeddings
    u_y = model.label_enc(y_labels, mask=mask)  # [..., embed_dim]
    
    # Split keys
    key_t, key_eps_sdm, key_eps_ce = jax.random.split(key, 3)
    
    # --- Shared components for SDM and CE loss parts ---
    # Sample continuous time t ~ U[0,1]
    # Ensure t_continuous has shape [batch_size, 1] for broadcasting with model.get_alpha_bar
    t_continuous_sdm = jax.random.uniform(key_t, (batch_size, 1))
    
    # --- 1. Score Diffusion Matching like loss (loss_sdm) ---
    # SNR'(t) = d(SNR(t))/dt where SNR(t) = ᾱ(t) / (1 - ᾱ(t))
    def snr_t(t):
        ab_t = model.get_alpha_bar(t)
        return jnp.squeeze(ab_t / (1 - ab_t))

    snr_t, snr_p = jax.vmap(jax.value_and_grad(snr_t))(t_continuous_sdm)
    
    # Get ᾱ(t) from the model's noise schedule
    axis = tuple(range(-len(u_y.shape[1:]), 0))
    ab_t = jnp.expand_dims(snr_t / (1 + snr_t), axis)

    eps_sdm = jax.random.normal(key_eps_sdm, u_y.shape)

    # zt = √ᾱ(t)*u_y + √(1-ᾱ(t))*ε
    zt_noisy_sdm = jnp.sqrt(ab_t) * u_y + jnp.sqrt(1.0 - ab_t) * eps_sdm
    
    # Get model's prediction for u_y from (x_imgs, zt_noisy_sdm, t_continuous_sdm)
    logits_sdm = model(x_imgs, zt_noisy_sdm, t_continuous_sdm, mask=mask)
    
    # pred_e = softmax(logits) @ W_embed (model.embed_matrix)
    probabilities_sdm = jax.nn.softmax(logits_sdm, axis=-1)
    pred_e_sdm = model.prob_enc(probabilities_sdm, mask=mask)
    
    # MSE loss for SDM part
    axis = tuple(range(- u_y.ndim + 1, 0))
    mse_sdm = jnp.sum((pred_e_sdm - u_y)**2, axis=axis)  # Sum over embed_dim
    diff = mse_sdm.ndim - snr_p.ndim
    if diff > 0:
        snr_p = jnp.expand_dims(snr_p, tuple(range(-diff, 0)))
    loss_sdm = 0.5 * eta * jnp.mean(snr_p * mse_sdm) # Mean over batch

    # --- 2. KL Divergence term (loss_kl) ---
    # D_KL(q(z_0|y) || p(z_0)) where z_0
    # This simplifies to 0.5 * ||u_y||^2 if p(z_0) ~ N(0,I)
    loss_kl = 0.5 * jnp.mean(jnp.sum(u_y**2, axis=axis))

    # --- 3. Cross-Entropy loss at t=1 (loss_ce) ---
    t_one = jnp.ones_like(t_continuous_sdm) # Time is 1.0
    alpha_bar_t1 = jax.vmap(model.get_alpha_bar)(t_one) # [batch_size, 1]
    diff = u_y.ndim - alpha_bar_t1.ndim
    if diff > 0:
        alpha_bar_t1 = jnp.expand_dims(alpha_bar_t1, tuple(range(-diff, 0)))
    
    eps_ce = jax.random.normal(key_eps_ce, u_y.shape)
    # z1_noisy = ᾱ(1)*u_y + √(1-ᾱ(1))*ε
    z1_noisy = jnp.sqrt(alpha_bar_t1) * u_y + jnp.sqrt(1.0 - alpha_bar_t1) * eps_ce
    
    logits_ce = model(x_imgs, z1_noisy, t_one, mask=mask)
    # Cross-entropy between predicted logits and true integer labels
    mask = mask.astype(jnp.bool) if mask is not None else None
    loss_ce = optax.softmax_cross_entropy_with_integer_labels(logits_ce, y_labels, where=mask)
    if loss_ce.ndim > 1:
        axis = tuple(range(- loss_ce.ndim + 1, 0))
        loss_ce = jnp.sum(loss_ce, axis=axis)
    loss_ce = jnp.mean(loss_ce)
    
    # --- Total Loss ---
    total_loss = loss_ce + loss_kl + loss_sdm
    if return_individual_losses:
        return total_loss, (loss_ce, loss_kl, loss_sdm)
    else:
        return total_loss, None

@eqx.filter_jit
def train_step(state: TrainState,
               x: jnp.ndarray,
               y: jnp.ndarray,
               optimizer: optax.GradientTransformation,
               eta: float = 1.0,
               mask: Optional[jnp.ndarray] = None,
               return_individual_losses: Optional[bool] = False) -> Tuple[TrainState, jnp.ndarray]:
    """ Single training step. """
    
    current_key, new_key = jax.random.split(state.key)
    
    (loss_val, losses), grads = eqx.filter_value_and_grad(compute_loss, has_aux=True)(
        state.model, x, y, current_key, eta, mask=mask, return_individual_losses=return_individual_losses)
    
    updates, new_optimizer_state = optimizer.update(grads, state.optimizer_state, state.model)
    new_model = eqx.apply_updates(state.model, updates)
    
    new_state = TrainState(
        model=new_model,
        optimizer_state=new_optimizer_state,
        key=new_key
    )

    if losses is None:
        result = loss_val
    else:
        result = (loss_val,) + losses
    return new_state, result

def create_train_state(model: NoPropCT,
                       optimizer: optax.GradientTransformation,
                       key: jax.random.PRNGKey
    ) -> TrainState:
    """ Create initial training state. """
    trainable_params = eqx.filter(model, eqx.is_inexact_array)
    optimizer_state = optimizer.init(trainable_params)
    
    return TrainState(
        model=model,
        optimizer_state=optimizer_state,
        key=key
    )
