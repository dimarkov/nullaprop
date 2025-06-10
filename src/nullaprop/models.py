"""
Core neural network models for NoProp implementation.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple, Optional
from jaxtyping import PRNGKeyArray, Array
import math
from collections.abc import Sequence

# ----------------------------------------------------------------------------
# Sinusoidal embedding for scalar t (JAX version)
# ----------------------------------------------------------------------------
def sinusoidal_embedding_jax(t: jnp.ndarray, dim: int) -> jnp.ndarray:
    """
    Generates sinusoidal embeddings for a batch of scalar time values.
    Args:
        t: Time tensor of shape [batch_size, 1] or [batch_size].
        dim: Embedding dimension.
    Returns:
        Sinusoidal embeddings of shape [batch_size, dim].
    """
    if t.ndim > 1:
        t = t[..., None] # Ensure t is [B, 1]
    half = dim // 2
    freqs = jnp.exp(
        -math.log(10000) * jnp.arange(half, dtype=t.dtype) / (half - 1)
    )
    args = t * freqs
    embedding = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)
    if dim % 2 == 1: # Handle odd dimensions
        pad_width = [(0, 0)] * embedding.ndim
        pad_width[-1] = [(0, 1)]
        embedding = jnp.pad(embedding, pad_width=pad_width)
    return embedding

class Conv2d(eqx.nn.Conv):
    """As [`equinox.nn.Conv`][] with `num_spatial_dims=2`."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = (1, 1),
        padding: str | int | Sequence[int] | Sequence[tuple[int, int]] = (0, 0),
        dilation: int | Sequence[int] = (1, 1),
        groups: int = 1,
        use_bias: bool = True,
        padding_mode: str = "ZEROS",
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        super().__init__(
            num_spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            use_bias=use_bias,
            padding_mode=padding_mode,
            dtype=dtype,
            key=key,
        )

    def __call__(self, x: Array, *, key: PRNGKeyArray | None = None) -> Array:
        """**Arguments:**

        - `x`: The input. Should be a JAX array of shape
            `(batch_dim, in_channels, dim_1, ..., dim_N)`, where `N = num_spatial_dims`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        A JAX array of shape `(batch_dim, out_channels, new_dim_1, ..., new_dim_N)`.
        """

        unbatched_rank = self.num_spatial_dims + 2
        if x.ndim != unbatched_rank:
            raise ValueError(
                f"Input to `Conv` needs to have rank {unbatched_rank},",
                f" but input has shape {x.shape}.",
            )

        padding = self.padding

        x = jax.lax.conv_general_dilated(
            lhs=x,
            rhs=self.weight,
            window_strides=self.stride,
            padding=padding,
            rhs_dilation=self.dilation,
            feature_group_count=self.groups,
        )

        if self.use_bias:
            x = x + self.bias
        return x

class CNN(eqx.Module):
    """CNN feature extractor for image inputs."""
    
    conv1: Conv2d
    conv2: Conv2d
    linear: eqx.nn.Linear
    
    def __init__(self, key: jax.random.PRNGKey, input_channels: int = 1, feature_dim: int = 128):
        key1, key2, key3 = jax.random.split(key, 3)
        
        self.conv1 = Conv2d(input_channels, 32, 3, padding=1, key=key1)
        self.conv2 = Conv2d(32, 64, 3, padding=1, key=key2)
        
        conv_output_size = 3136 if input_channels == 1 else 4096 # MNIST vs CIFAR
        self.linear = eqx.nn.Linear(conv_output_size, feature_dim, key=key3)
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B = x.shape[0]
        x_out = jax.nn.relu(self.conv1(x))
        x_out = jax.lax.reduce_window(x_out, -jnp.inf, jax.lax.max, (1, 1, 2, 2), (1, 1, 2, 2), 'VALID')
        x_out = jax.nn.relu(self.conv2(x_out))
        x_out = jax.lax.reduce_window(x_out, -jnp.inf, jax.lax.max, (1, 1, 2, 2), (1, 1, 2, 2), 'VALID')
        x_out = x_out.reshape(B, -1)
        return jax.vmap(self.linear)(x_out)

class LabelEncoder(eqx.Module):
    """Encodes a label-embedding vector z_t via a small FC net with skip connection."""
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(self, embed_dim: int, *, key: jax.random.PRNGKey):
        k1, k2 = jax.random.split(key)
        hidden_dim = embed_dim  # As in PyTorch version
        self.fc1 = eqx.nn.Linear(embed_dim, hidden_dim, key=k1)
        self.fc2 = eqx.nn.Linear(hidden_dim, embed_dim, key=k2)

    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        out = self.fc2(jax.nn.relu(self.fc1(z)))
        return out + z

class TimeEncoder(eqx.Module):
    """Encodes a timestamp t (shape [B,1]) into embedding (shape [B, embed_dim])."""
    fc: eqx.nn.Linear
    time_emb_dim_internal: int # The dimension of the raw sinusoidal embedding

    def __init__(self, time_emb_dim_internal: int, embed_dim: int, *, key: jax.random.PRNGKey):
        self.time_emb_dim_internal = time_emb_dim_internal
        self.fc = eqx.nn.Linear(time_emb_dim_internal, embed_dim, key=key)

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        # t: [B,1] -> sinusoidal [B, time_emb_dim_internal]
        te = sinusoidal_embedding_jax(t, self.time_emb_dim_internal)
        return jax.nn.relu(self.fc(te)) # Apply ReLU as in PyTorch version's Sequential

class FuseHead(eqx.Module):
    """Combines image, z, and t features, and outputs class logits."""
    net: eqx.nn.Sequential

    def __init__(self, embed_dim: int, mid_dim: int, num_classes: int, *, key: jax.random.PRNGKey):
        k1, k2, k3 = jax.random.split(key, 3)
        # Using LayerNorm instead of BatchNorm for simplicity in Equinox state management
        self.net = eqx.nn.Sequential([
            eqx.nn.Linear(embed_dim * 3, embed_dim, key=k1),
            eqx.nn.LayerNorm(embed_dim),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Linear(embed_dim, mid_dim, key=k2),
            eqx.nn.LayerNorm(mid_dim),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Linear(mid_dim, num_classes, key=k3)
        ])

    def __call__(self, fx: jnp.ndarray, fz: jnp.ndarray, ft: jnp.ndarray) -> jnp.ndarray:
        x_cat = jnp.concatenate([fx, fz, ft], axis=-1)
        return self.net(x_cat)

class NoiseSchedule(eqx.Module):
    """Learnable noise schedule module."""
    gamma_tilde_net: eqx.Module
    gamma0: jnp.ndarray
    gamma1: jnp.ndarray

    def __init__(self, hidden_dim: int = 64, *, key: jax.random.PRNGKey):
        k1, k2 = jax.random.split(key, 2)
        self.gamma_tilde_net = eqx.nn.Sequential([
            eqx.nn.Linear(1, hidden_dim, key=k1),
            eqx.nn.Lambda(jax.nn.softplus),
            eqx.nn.Linear(hidden_dim, 1, key=k2),
            eqx.nn.Lambda(jax.nn.softplus)
        ])

        self.gamma0 = jnp.array(-5.0)
        self.gamma1 = jnp.array(5.0)

    def _gamma_bar(self, t: jnp.ndarray) -> jnp.ndarray:
        g0 = self.gamma_tilde_net(jnp.zeros_like(t))
        g1 = self.gamma_tilde_net(jnp.ones_like(t))
        return jnp.clip((self.gamma_tilde_net(t) - g0) / (g1 - g0 + 1e-8), 0.0, 1.0)

    def alpha_bar(self, t: jnp.ndarray) -> jnp.ndarray:
        # t shape (1,)
        gamma_val_t = self.gamma0 + (self.gamma1 - self.gamma0) * (1.0 - self._gamma_bar(t))
        return jnp.clip(jax.nn.sigmoid(- gamma_val_t / 2.0), 1e-5, 1-1e-5)


class NoPropCT(eqx.Module):
    """
    NoProp continuous time (CT) model aligned with yhgon/NoProp (PyTorch) structure.
    """
    cnn: CNN
    label_enc: LabelEncoder
    time_enc: TimeEncoder
    fuse_head: FuseHead
    noise_schedule: NoiseSchedule
    
    embed_matrix: jnp.ndarray  # Corresponds to W_embed: (num_classes, embed_dim)
    
    # Dimensions needed for initialization and potentially by other modules
    num_classes: int
    embed_dim: int 

    def __init__(
        self, 
        *,
        key: jax.random.PRNGKey,
        num_classes: int,
        embed_dim: int, # Default from yhgon/NoProp argparse for embed_dim
        mid_dim: int = 128, # Default CNN output
        input_channels: int = 1,
        time_emb_dim_internal: int = 64, # Default from yhgon/NoProp argparse for time_emb_dim
        noise_schedule_hidden_dim: int = 64,
    ):
        k_cnn, k_label, k_time, k_fuse, k_noise, k_embed_matrix = jax.random.split(key, 6)
        
        self.cnn = CNN(key=k_cnn, input_channels=input_channels, feature_dim=embed_dim)
        self.label_enc = LabelEncoder(embed_dim=embed_dim, key=k_label)
        # The embed_dim for TimeEncoder is the target dimension after FC layer, matching other embed_dims
        self.time_enc = TimeEncoder(time_emb_dim_internal=time_emb_dim_internal, embed_dim=embed_dim, key=k_time)
        self.fuse_head = FuseHead(embed_dim=embed_dim, mid_dim=mid_dim, num_classes=num_classes, key=k_fuse)
        self.noise_schedule = NoiseSchedule(hidden_dim=noise_schedule_hidden_dim, key=k_noise)

        self.embed_matrix = jax.random.normal(k_embed_matrix, (num_classes, embed_dim)) * 0.01

        self.num_classes = num_classes
        self.embed_dim = embed_dim

    def get_alpha_bar(self, t: jnp.ndarray) -> jnp.ndarray:
        """Convenience method to get alpha_bar from the noise schedule."""
        return self.noise_schedule.alpha_bar(t)

    def __call__(self, x_imgs: jnp.ndarray, z_label_embedding_noisy: jnp.ndarray, t_continuous: jnp.ndarray) -> jnp.ndarray:
        """
        Unified forward pass through the model components.
        Args:
            x_imgs: Input images [batch_size, channels, height, width]
            z_label_embedding_noisy: Noisy label embeddings [batch_size, embed_dim]
            t_continuous: Continuous time values [batch_size, 1]
        Returns:
            Logits from the FuseHead [batch_size, num_classes]
        """
        fx = self.cnn(x_imgs)  # [embed_dim]
        fz = jax.vmap(self.label_enc)(z_label_embedding_noisy)  # [embed_dim]
        ft = jax.vmap(self.time_enc)(t_continuous)  # [embed_dim]

        return jax.vmap(self.fuse_head)(fx, fz, ft)


def init_noprop_model(
    key: jax.random.PRNGKey,
    num_classes: int = 10,
    embed_dim: int = 256,
    mid_dim: int = 128, 
    input_channels: int = 1,
    time_emb_dim_internal: int = 64, 
    noise_schedule_hidden_dim: int = 64,
) -> NoPropCT:
    """
    Initialize a NoProp model with parameters aligned with yhgon/NoProp.
    Note: The CNN's output feature_dim is implicitly set to `embed_dim` here
    to match the expectation of the FuseHead.
    """
    return NoPropCT(
        key=key,
        num_classes=num_classes,
        embed_dim=embed_dim,
        mid_dim=mid_dim,
        input_channels=input_channels,
        time_emb_dim_internal=time_emb_dim_internal,
        noise_schedule_hidden_dim=noise_schedule_hidden_dim,
    )
