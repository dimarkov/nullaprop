"""
Core neural network models for NoProp implementation.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple, Optional
import math


class CNN(eqx.Module):
    """CNN feature extractor for image inputs."""
    
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    linear: eqx.nn.Linear
    
    def __init__(self, key: jax.random.PRNGKey, input_channels: int = 1, feature_dim: int = 128):
        """
        Initialize CNN feature extractor.
        
        Args:
            key: Random key for initialization
            input_channels: Number of input channels (1 for MNIST, 3 for CIFAR)
            feature_dim: Output feature dimension
        """
        key1, key2, key3 = jax.random.split(key, 3)
        
        self.conv1 = eqx.nn.Conv2d(input_channels, 32, 3, padding=1, key=key1)
        self.conv2 = eqx.nn.Conv2d(32, 64, 3, padding=1, key=key2)
        
        # For MNIST (28x28): after 2 maxpools -> 7x7x64 = 3136
        # For CIFAR (32x32): after 2 maxpools -> 8x8x64 = 4096
        conv_output_size = 3136 if input_channels == 1 else 4096
        self.linear = eqx.nn.Linear(conv_output_size, feature_dim, key=key3)
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through CNN.
        
        Args:
            x: Input images [batch_size, channels, height, width]
            
        Returns:
            Feature vectors [batch_size, feature_dim]
        """
        # Define single-sample forward pass
        def forward_single(x_single):
            # x_single has shape (channels, height, width)
            # Conv1 + ReLU + MaxPool
            x_out = jax.nn.relu(self.conv1(x_single))
            x_out = jax.lax.reduce_window(
                x_out, -jnp.inf, jax.lax.max, 
                window_dimensions=(1, 2, 2), 
                window_strides=(1, 2, 2), 
                padding='VALID'
            )
            
            # Conv2 + ReLU + MaxPool  
            x_out = jax.nn.relu(self.conv2(x_out))
            x_out = jax.lax.reduce_window(
                x_out, -jnp.inf, jax.lax.max,
                window_dimensions=(1, 2, 2),
                window_strides=(1, 2, 2),
                padding='VALID'
            )
            
            # Flatten and linear
            x_out = x_out.reshape(-1)  # Flatten to 1D
            x_out = self.linear(x_out)
            
            return x_out
        
        # Apply over batch dimension using vmap
        return jax.vmap(forward_single)(x)


def apply_mlp(mlp_params: jnp.ndarray, x_features: jnp.ndarray, z_noisy: jnp.ndarray) -> jnp.ndarray:
    """
    Apply MLP with given parameters to combined input.
    
    Args:
        mlp_params: Flattened MLP parameters
        x_features: Image features [batch_size, feature_dim]
        z_noisy: Noisy label embeddings [batch_size, embed_dim]
        
    Returns:
        Predicted clean label embeddings [batch_size, embed_dim]
    """
    # Combine features
    combined = jnp.concatenate([x_features, z_noisy], axis=-1)
    batch_size = combined.shape[0]
    input_dim = combined.shape[1]
    embed_dim = z_noisy.shape[-1]
    
    # Infer hidden_dim from parameter count
    # Total params = input_dim * hidden_dim + hidden_dim + hidden_dim * embed_dim + embed_dim
    # Total params = hidden_dim * (input_dim + 1 + embed_dim) + embed_dim
    # hidden_dim = (total_params - embed_dim) / (input_dim + 1 + embed_dim)
    total_params = mlp_params.shape[0]
    hidden_dim = (total_params - embed_dim) // (input_dim + 1 + embed_dim)
    
    # Calculate parameter splits
    w1_size = input_dim * hidden_dim
    b1_size = hidden_dim
    w2_size = hidden_dim * embed_dim
    b2_size = embed_dim
    
    # Extract weights and biases
    W1 = mlp_params[:w1_size].reshape(input_dim, hidden_dim)
    b1 = mlp_params[w1_size:w1_size + b1_size]
    W2 = mlp_params[w1_size + b1_size:w1_size + b1_size + w2_size].reshape(hidden_dim, embed_dim)
    b2 = mlp_params[w1_size + b1_size + w2_size:w1_size + b1_size + w2_size + b2_size]
    
    # Forward pass
    hidden = jax.nn.relu(combined @ W1 + b1)
    output = hidden @ W2 + b2
    
    return output


def calculate_mlp_param_size(feature_dim: int, embed_dim: int, hidden_dim: int = 256) -> int:
    """Calculate total number of parameters for MLP."""
    input_dim = feature_dim + embed_dim
    
    # Layer 1: input_dim -> hidden_dim
    w1_params = input_dim * hidden_dim
    b1_params = hidden_dim
    
    # Layer 2: hidden_dim -> embed_dim
    w2_params = hidden_dim * embed_dim
    b2_params = embed_dim
    
    return w1_params + b1_params + w2_params + b2_params


class NoPropModel(eqx.Module):
    """
    NoProp model with CNN feature extractor and T denoising MLPs.
    """
    
    cnn: CNN
    mlp_params: jnp.ndarray  # Shape: (T, param_size)
    alpha_schedule: jnp.ndarray  # Shape: (T,)
    embed_matrix: jnp.ndarray  # Shape: (num_classes, embed_dim)
    classifier: eqx.nn.Linear  # Final classification layer
    T: int
    embed_dim: int
    feature_dim: int
    num_classes: int
    
    def __init__(
        self, 
        key: jax.random.PRNGKey,
        T: int = 10,
        num_classes: int = 10,
        embed_dim: int = 10,
        feature_dim: int = 128,
        input_channels: int = 1,
        hidden_dim: int = 256,
        embedding_type: str = "one_hot"
    ):
        """
        Initialize NoProp model.
        
        Args:
            key: Random key for initialization
            T: Number of diffusion steps
            num_classes: Number of output classes
            embed_dim: Dimension of label embeddings
            feature_dim: CNN output feature dimension
            input_channels: Number of input channels
            hidden_dim: MLP hidden dimension
            embedding_type: Type of embedding ("one_hot", "learnable", "prototype")
        """
        key_cnn, key_mlp, key_embed, key_classifier = jax.random.split(key, 4)
        
        # Initialize CNN
        self.cnn = CNN(key_cnn, input_channels, feature_dim)
        
        # Calculate MLP parameter size
        param_size = calculate_mlp_param_size(feature_dim, embed_dim, hidden_dim)
        
        # Initialize T sets of MLP parameters
        self.mlp_params = jax.random.normal(key_mlp, (T, param_size)) * 0.1
        
        # Create cosine noise schedule as in the paper
        self.alpha_schedule = self._create_cosine_schedule(T)
        
        # Initialize embedding matrix
        if embedding_type == "one_hot":
            # Fixed one-hot embeddings
            if embed_dim == num_classes:
                self.embed_matrix = jnp.eye(num_classes)
            else:
                # Pad or truncate to match embed_dim
                eye_matrix = jnp.eye(num_classes)
                if embed_dim > num_classes:
                    # Pad with zeros
                    padding = jnp.zeros((num_classes, embed_dim - num_classes))
                    self.embed_matrix = jnp.concatenate([eye_matrix, padding], axis=1)
                else:
                    # Truncate
                    self.embed_matrix = eye_matrix[:, :embed_dim]
        else:
            # Learnable embeddings (random or orthogonal initialization)
            if embed_dim == num_classes:
                # Initialize as orthogonal matrix
                from scipy.stats import ortho_group
                self.embed_matrix = jnp.array(ortho_group.rvs(num_classes, random_state=0))
            else:
                # Random initialization
                self.embed_matrix = jax.random.normal(key_embed, (num_classes, embed_dim)) * 0.1
        
        # Initialize classifier
        self.classifier = eqx.nn.Linear(embed_dim, num_classes, key=key_classifier)
        
        # Store dimensions
        self.T = T
        self.embed_dim = embed_dim
        self.feature_dim = feature_dim
        self.num_classes = num_classes
    
    def _create_cosine_schedule(self, T: int) -> jnp.ndarray:
        """Create cosine noise schedule as in the paper."""
        s = 0.008  # Small offset
        steps = jnp.arange(T + 1, dtype=jnp.float32) / T
        alphas_cumprod = jnp.cos((steps + s) / (1 + s) * jnp.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        alphas = jnp.concatenate([jnp.array([1.0]), alphas_cumprod[1:] / alphas_cumprod[:-1]])
        return jnp.clip(alphas[1:], 0.0001, 0.9999)  # Exclude first element, clip for stability
    
    def extract_features(self, x: jnp.ndarray) -> jnp.ndarray:
        """Extract features using CNN."""
        return self.cnn(x)
    
    def apply_single_mlp(self, t: int, x_features: jnp.ndarray, z_noisy: jnp.ndarray) -> jnp.ndarray:
        """
        Apply MLP for a specific timestep t.
        
        Args:
            t: Timestep index
            x_features: Image features [batch_size, feature_dim]
            z_noisy: Noisy label embeddings [batch_size, embed_dim]
            
        Returns:
            Predicted clean label embeddings [batch_size, embed_dim]
        """
        return apply_mlp(self.mlp_params[t], x_features, z_noisy)
    
    def denoise_parallel(self, x_features: jnp.ndarray, z_noisy_all: jnp.ndarray) -> jnp.ndarray:
        """
        Apply all T denoising MLPs in parallel.
        
        Args:
            x_features: Image features [batch_size, feature_dim]
            z_noisy_all: Noisy embeddings for all steps [T, batch_size, embed_dim]
            
        Returns:
            Predicted clean embeddings [T, batch_size, embed_dim]
        """
        # Use vmap to apply different MLP parameters to different time steps
        @eqx.filter_vmap(in_axes=(0, None, 0))  # vmap over T (first axis)
        def apply_layer(layer_params, x_feat, z_noisy):
            return apply_mlp(layer_params, x_feat, z_noisy)
        
        return apply_layer(self.mlp_params, x_features, z_noisy_all)


def init_noprop_model(
    key: jax.random.PRNGKey,
    T: int = 10,
    num_classes: int = 10,
    embed_dim: int = 10,
    feature_dim: int = 128,
    input_channels: int = 1,
    hidden_dim: int = 256,
    embedding_type: str = "one_hot"
) -> NoPropModel:
    """
    Initialize a NoProp model with default parameters.
    
    Args:
        key: Random key for initialization
        T: Number of diffusion steps
        num_classes: Number of output classes
        embed_dim: Dimension of label embeddings
        feature_dim: CNN feature dimension
        input_channels: Input channels (1 for MNIST, 3 for CIFAR)
        hidden_dim: MLP hidden dimension
        embedding_type: Type of embedding ("one_hot", "learnable", "prototype")
        
    Returns:
        Initialized NoPropModel
    """
    return NoPropModel(
        key=key,
        T=T,
        num_classes=num_classes,
        embed_dim=embed_dim,
        feature_dim=feature_dim,
        input_channels=input_channels,
        hidden_dim=hidden_dim,
        embedding_type=embedding_type
    )
