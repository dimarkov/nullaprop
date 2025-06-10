"""
Inference utilities for NoProp implementation.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Tuple
from .models import NoPropCT


@eqx.filter_jit
def inference_ct_euler(
    model: NoPropCT, 
    x_imgs: jnp.ndarray, 
    key: jax.random.PRNGKey, 
    T_steps: int = 1000
) -> jnp.ndarray:
    """
    Perform inference using the Euler method for the reverse ODE,
    aligned with `run_noprop_ct_inference` from yhgon/NoProp.

    Args:
        model: Trained NoPropModel instance.
        x_imgs: Input images [batch_size, channels, height, width].
        key: JAX random key for initial noise.
        T_steps: Number of discretization steps for the ODE solver.
        
    Returns:
        Predicted class labels [batch_size].
    """
    batch_size = x_imgs.shape[0]
    embed_dim = model.embed_dim 
    
    dt = 1.0 / T_steps
    
    # Initial z at t=0 is pure noise N(0,I)
    z = jax.random.normal(key, (batch_size, embed_dim))
    
    # Define the loop body for jax.lax.fori_loop
    def euler_step_body(i, current_z):
        # Current time t = i / T_steps
        t = jnp.full((batch_size, 1), i / T_steps)
        
        # Get ᾱ(t) from model's noise schedule
        ab_t = jax.vmap(model.get_alpha_bar)(t) # [batch_size, 1]
        
        # Get logits: model.forward_unified(x_imgs, current_z, t)
        logits = model(x_imgs, current_z, t)
        
        # Probabilities and predicted clean embedding
        probabilities = jax.nn.softmax(logits, axis=1)
        pred_e = probabilities @ model.embed_matrix # [batch_size, embed_dim]
        
        # Euler update: z_{t+dt} = z_t + dt * f(z_t, t)
        # where f(z, t) = (pred_e(z,t) - z) / (1 - ᾱ(t))
        dz_dt = (pred_e - current_z) / (1.0 - ab_t)
        next_z = current_z + dt * dz_dt
        return next_z

    # Run the loop
    z_final_approx_at_t1 = jax.lax.fori_loop(0, T_steps, euler_step_body, z)
    
    # Final prediction at t=1 using the evolved z
    t_one = jnp.ones((batch_size, 1))
    final_logits = model(x_imgs, z_final_approx_at_t1, t_one)
    
    return jnp.argmax(final_logits, axis=-1)


@eqx.filter_jit
def inference_ct_heun(
    model: NoPropCT, 
    x_imgs: jnp.ndarray, 
    key: jax.random.PRNGKey, # key is for initial z, not used in loop if deterministic
    T_steps: int = 40
) -> jnp.ndarray:
    """
    Perform inference using the Heun method for the reverse ODE,
    aligned with `run_noprop_ct_inference_heun` from yhgon/NoProp.

    Args:
        model: Trained NoPropModel instance.
        x_imgs: Input images [batch_size, channels, height, width].
        key: JAX random key for initial noise.
        T_steps: Number of discretization steps for the ODE solver.
        
    Returns:
        Predicted class labels [batch_size].
    """
    batch_size = x_imgs.shape[0]
    embed_dim = model.embed_dim
    dt = 1.0 / T_steps

    z = jax.random.normal(key, (batch_size, embed_dim)) # Initial z ~ N(0,I)

    def heun_step_body(i, current_z):
        t_n = jnp.full((batch_size, 1), i / T_steps)
        t_np1 = jnp.full((batch_size, 1), (i + 1) / T_steps)

        alpha_n = jax.vmap(model.get_alpha_bar)(t_n)
        logits_n = model(x_imgs, current_z, t_n)
        p_n = jax.nn.softmax(logits_n, axis=-1)
        pred_n = p_n @ model.embed_matrix
        f_n = (pred_n - current_z) / (1.0 - alpha_n)

        # Predictor step (Euler)
        z_mid_pred = current_z + dt * f_n
        
        # Corrector step
        alpha_m = jax.vmap(model.get_alpha_bar)(t_np1) # alpha at t_{n+1} for z_mid
        logits_mid = model(x_imgs, z_mid_pred, t_np1)
        p_mid = jax.nn.softmax(logits_mid, axis=-1)
        pred_mid = p_mid @ model.embed_matrix
        f_mid = (pred_mid - z_mid_pred) / (1.0 - alpha_m)
        
        next_z = current_z + 0.5 * dt * (f_n + f_mid)
        return next_z

    z_final_approx_at_t1 = jax.lax.fori_loop(0, T_steps, heun_step_body, z)

    t_one = jnp.ones((batch_size, 1))
    final_logits = model(x_imgs, z_final_approx_at_t1, t_one)
    return jnp.argmax(final_logits, axis=-1)
