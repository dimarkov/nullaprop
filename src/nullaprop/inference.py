"""
Inference utilities for NoProp implementation.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Tuple
from .models import NoPropCT

def propagate(pred, z, abar, abarder, fix):
    if fix:
        return (jnp.sqrt(abar) * pred - (1 + abar) * z / 2) * abarder / (abar * (1 - abar))
    else:
        return (pred - z) / (1.0 - abar)

@eqx.filter_jit
def inference_ct_euler(
    model: NoPropCT, 
    x_imgs: jnp.ndarray, 
    key: jax.random.PRNGKey, 
    T_steps: int = 1000,
    fix: bool = False
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
        ab_t, ab_grad_t = jax.vmap(
            jax.value_and_grad(lambda x: model.get_alpha_bar(x).squeeze(-1))
        )(t)
        ab_t = jnp.expand_dims(ab_t, -1) 
        
        # Get logits: model.forward_unified(x_imgs, current_z, t)
        logits = model(x_imgs, current_z, t)
        
        # Probabilities and predicted clean embedding
        probabilities = jax.nn.softmax(logits, axis=1)
        pred_e = model.prob_enc(probabilities)
        
        # Euler update: z_{t+dt} = z_t + dt * f(z_t, t)
        # where f(z, t) = (pred_e(z,t) - z) / (1 - ᾱ(t))
        dz_dt = propagate(pred_e, current_z, ab_t, ab_grad_t, fix)
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
    T_steps: int = 40,
    shape: tuple = None,
    fix: bool = False
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
    shape = x_imgs.shape[:1] if shape is None else shape
    embed_dim = model.embed_dim
    dt = 1.0 / T_steps

    z = jax.random.normal(key, shape + (embed_dim,)) # Initial z ~ N(0,I)

    def heun_step_body(i, current_z):
        t_n = jnp.full((current_z.shape[0], 1), i / T_steps)
        t_np1 = jnp.full((current_z.shape[0], 1), (i + 1) / T_steps)

        diff = len(current_z.shape) - len(t_n.shape)
        alpha_n, alpha_grad_n = jax.vmap(
            jax.value_and_grad(lambda x: model.get_alpha_bar(x).squeeze(-1))
        )(t_n)
        alpha_n = jnp.expand_dims(alpha_n, -1) 
        if diff > 0:
            alpha_n = jnp.expand_dims(alpha_n, tuple(range(-diff, 0)))
            alpha_grad_n = jnp.expand_dims(alpha_grad_n, tuple(range(-diff, 0)))
        
        logits_n = model(x_imgs, current_z, t_n)
        p_n = jax.nn.softmax(logits_n, axis=-1)
        pred_n = model.prob_enc(p_n)
        f_n = propagate(pred_n, current_z, alpha_n, alpha_grad_n, fix)

        # Predictor step (Euler)
        z_mid_pred = current_z + dt * f_n
        
        # Corrector step
        alpha_m, alpha_grad_m = jax.vmap(
            jax.value_and_grad(lambda x: model.get_alpha_bar(x).squeeze(-1))
        )(t_np1) # alpha at t_{n+1} for z_mid
        alpha_m = jnp.expand_dims(alpha_m, -1)
        if diff > 0:
            alpha_m = jnp.expand_dims(alpha_m, tuple(range(-diff, 0)))
            alpha_grad_m = jnp.expand_dims(alpha_grad_m, tuple(range(-diff, 0)))
        
        logits_mid = model(x_imgs, z_mid_pred, t_np1)
        p_mid = jax.nn.softmax(logits_mid, axis=-1)
        pred_mid = model.prob_enc(p_mid)
        f_mid = propagate(pred_mid, z_mid_pred, alpha_grad_n, alpha_grad_m, fix)
        
        next_z = current_z + 0.5 * dt * (f_n + f_mid)
        return next_z

    z_final_approx_at_t1 = jax.lax.fori_loop(0, T_steps, heun_step_body, z)

    t_one = jnp.ones((z_final_approx_at_t1.shape[0], 1))
    final_logits = model(x_imgs, z_final_approx_at_t1, t_one)
    return jnp.argmax(final_logits, axis=-1)
