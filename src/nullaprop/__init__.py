"""
NoProp: Training Neural Networks Without Back-propagation or Forward-propagation
JAX/Equinox Implementation

A novel training method using diffusion-based denoising for independent layer training.
"""

from .models import NoPropModel, CNN, init_noprop_model
from .training import train_step, create_train_state, TrainState
from .inference import inference_step, inference_step_deterministic
from .utils import (
    load_mnist_data, load_cifar10_data, load_cifar100_data,
    create_noise_schedule, get_dataset_info, print_model_summary,
    evaluate_model, plot_training_curves
)
from .experiments import (
    run_mnist_experiment, run_cifar10_experiment,
    demonstrate_diffusion_process, demonstrate_inference_process,
    benchmark_performance
)

__version__ = "0.1.0"
__all__ = [
    # Core models
    "NoPropModel", 
    "CNN",
    "init_noprop_model",
    
    # Training
    "train_step", 
    "create_train_state",
    "TrainState",
    
    # Inference
    "inference_step",
    "inference_step_deterministic",
    
    # Data utilities
    "load_mnist_data",
    "load_cifar10_data", 
    "load_cifar100_data",
    "create_noise_schedule",
    "get_dataset_info",
    
    # Evaluation and visualization
    "print_model_summary",
    "evaluate_model",
    "plot_training_curves",
    
    # Experiments
    "run_mnist_experiment",
    "run_cifar10_experiment", 
    "demonstrate_diffusion_process",
    "demonstrate_inference_process",
    "benchmark_performance"
]
