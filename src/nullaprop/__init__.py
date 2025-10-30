"""
NoProp: Training Neural Networks Without Back-propagation or Forward-propagation
JAX/Equinox Implementation

A novel training method using diffusion-based denoising for independent layer training.
"""

from .models import NoPropCT, CNN, init_noprop_model, LabelEncoder, TimeEncoder, FuseHead, NoiseSchedule, sinusoidal_embedding
from .training import train_step, create_train_state, TrainState, compute_loss
from .inference import inference_ct_euler, inference_ct_heun
from .utils import (
    load_data, get_dataset_info, print_model_summary, initialize_with_prototypes_jax,
    evaluate_model, plot_training_curves
)
from .experiments import (
    run_experiment, benchmark_performance
)

__version__ = "0.1.0"
__all__ = [
    # Core models
    "NoPropCT",
    "CNN",
    "init_noprop_model",
    "LabelEncoder",
    "TimeEncoder",
    "FuseHead",
    "NoiseSchedule",
    "sinusoidal_embedding",
    
    # Training
    "train_step",
    "create_train_state",
    "TrainState",
    "compute_loss",
    
    # Inference
    "inference_ct_euler",
    "inference_ct_heun",
    
    # Data utilities
    "load_data",
    "get_dataset_info",
    "initialize_with_prototypes_jax",
    
    # Evaluation and visualization
    "print_model_summary",
    "evaluate_model",
    "plot_training_curves",
    
    # Experiments
    "run_experiment",
    "benchmark_performance"
]
