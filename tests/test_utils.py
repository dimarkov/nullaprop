"""
Pytest tests for utility functions in src/nullaprop/utils.py.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from unittest import mock
import io
import contextlib
import equinox as eqx

# Attempt to import from nullaprop.utils. If it's part of a package, this should work.
# If running tests directly in a way that src is not in PYTHONPATH, adjustments might be needed.
from nullaprop import utils
from nullaprop.models import init_noprop_model # For print_model_summary and evaluate_model tests

# --- Fixtures ---
@pytest.fixture
def key():
    return jax.random.PRNGKey(0)

# --- Tests for Preprocessing Functions ---

def test_preprocess_mnist_sample_normalization_and_shape():
    """Test MNIST sample preprocessing for normalization and shape."""
    # Create a mock PIL Image-like object (as a numpy array, since PIL is not directly used in the function)
    mock_image_data = np.random.randint(0, 256, (28, 28), dtype=np.uint8)
    sample = {'image': mock_image_data, 'label': 5}
    
    processed_sample = utils.preprocess_mnist_sample(sample)
    
    img = processed_sample['image']
    assert img.shape == (1, 28, 28)  # Expect [C, H, W] with C=1
    assert processed_sample['label'] == 5
    # Check if normalization happened (mean around 0, std around 1)
    # These are rough checks as exact values depend on the input distribution
    assert -2.0 < img.mean() < 2.0 # MNIST mean is 0.1307, std 0.3081. After (x-mean)/std, new mean is ~0
    assert 0.5 < img.std() < 2.0 # New std is ~1

def test_preprocess_cifar10_sample_normalization_and_shape():
    """Test CIFAR-10 sample preprocessing."""
    mock_image_data = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    sample = {'img': mock_image_data, 'label': 3}
    
    processed_sample = utils.preprocess_cifar10_sample(sample)
    
    img = processed_sample['image']
    assert img.shape == (3, 32, 32)  # Expect [C, H, W]
    assert processed_sample['label'] == 3
    assert -2.0 < img.mean() < 2.0 # CIFAR-10 means are ~0.5. After (x-mean)/std, new mean is ~0
    assert 0.5 < img.std() < 2.0 # New std is ~1

def test_preprocess_cifar100_sample_normalization_and_shape():
    """Test CIFAR-100 sample preprocessing."""
    mock_image_data = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    sample = {'img': mock_image_data, 'fine_label': 78, 'coarse_label': 10}
        
    processed_sample = utils.preprocess_cifar100_sample(sample)
    
    img = processed_sample['image']
    assert img.shape == (3, 32, 32)
    assert processed_sample['label'] == 78 # Uses fine_label
    assert -2.0 < img.mean() < 2.0
    assert 0.5 < img.std() < 2.0

# --- Tests for Data Loading ---

@mock.patch('nullaprop.utils.jdl.DataLoader')
@mock.patch('nullaprop.utils.jdl.ArrayDataset')
@mock.patch('nullaprop.utils.load_dataset')
def test_load_data_mnist(mock_load_dataset, MockArrayDataset, MockDataLoader):
    """Test load_data for MNIST dataset."""
    mock_hf_dataset = mock.MagicMock()
    mock_mapped_train = mock.MagicMock()
    mock_mapped_test = mock.MagicMock()
    
    # Simulate the structure returned by Hugging Face .map()
    mock_mapped_train.__getitem__.side_effect = lambda key: {'image': np.random.rand(10, 1, 28, 28), 'label': np.random.randint(0,9,10)}[key]
    mock_mapped_test.__getitem__.side_effect = lambda key: {'image': np.random.rand(5, 1, 28, 28), 'label': np.random.randint(0,9,5)}[key]

    mock_hf_dataset.__getitem__.side_effect = lambda key: {'train': mock_mapped_train, 'test': mock_mapped_test}[key]
    mock_hf_dataset.with_format.return_value = mock_hf_dataset # chain with_format
    
    # Simulate the .map() call
    mock_hf_dataset['train'].map.return_value = mock_mapped_train
    mock_hf_dataset['test'].map.return_value = mock_mapped_test
    
    mock_load_dataset.return_value = mock_hf_dataset

    train_ds, test_ds = utils.load_data('mnist', batch_size=64)

    mock_load_dataset.assert_called_once_with('mnist', cache_dir="./data")
    mock_hf_dataset['train'].map.assert_called_once_with(utils.preprocess_mnist_sample, batched=True, remove_columns=['image', 'label'])
    mock_hf_dataset['test'].map.assert_called_once_with(utils.preprocess_mnist_sample, batched=True, remove_columns=['image', 'label'])
    
    assert MockArrayDataset.call_count == 2
    assert MockDataLoader.call_count == 2
    # Check train DataLoader args
    MockDataLoader.assert_any_call(MockArrayDataset.return_value, 'jax', batch_size=64, drop_last=True, shuffle=True)
    # Check test DataLoader args
    MockDataLoader.assert_any_call(MockArrayDataset.return_value, 'jax', batch_size=10_000, shuffle=False)


@mock.patch('nullaprop.utils.jdl.DataLoader')
@mock.patch('nullaprop.utils.jdl.ArrayDataset')
@mock.patch('nullaprop.utils.load_dataset')
def test_load_data_cifar10(mock_load_dataset, MockArrayDataset, MockDataLoader):
    """Test load_data for CIFAR-10 dataset."""
    mock_hf_dataset = mock.MagicMock()
    mock_mapped_train = mock.MagicMock()
    mock_mapped_test = mock.MagicMock()
    
    mock_mapped_train.__getitem__.side_effect = lambda key: {'image': np.random.rand(10, 3, 32, 32), 'label': np.random.randint(0,9,10)}[key]
    mock_mapped_test.__getitem__.side_effect = lambda key: {'image': np.random.rand(5, 3, 32, 32), 'label': np.random.randint(0,9,5)}[key]

    mock_hf_dataset.__getitem__.side_effect = lambda key: {'train': mock_mapped_train, 'test': mock_mapped_test}[key]
    mock_hf_dataset.with_format.return_value = mock_hf_dataset
    
    mock_hf_dataset['train'].map.return_value = mock_mapped_train
    mock_hf_dataset['test'].map.return_value = mock_mapped_test
    
    mock_load_dataset.return_value = mock_hf_dataset

    utils.load_data('cifar10', batch_size=128)

    mock_load_dataset.assert_called_once_with('cifar10', cache_dir="./data")
    mock_hf_dataset['train'].map.assert_called_once_with(utils.preprocess_cifar10_sample, batched=True, remove_columns=['img', 'label'])
    mock_hf_dataset['test'].map.assert_called_once_with(utils.preprocess_cifar10_sample, batched=True, remove_columns=['img', 'label'])


@mock.patch('nullaprop.utils.load_dataset') # Mock the call to Hugging Face load_dataset
def test_load_data_unsupported(mock_hf_load_dataset):
    """Test load_data for an unsupported dataset name when the name is not in its internal if/else."""
    # Configure the mock to return a dummy dataset for "unsupported_dataset",
    # so that the function proceeds to its own if/else logic.
    mock_dummy_dataset = mock.MagicMock()
    mock_dummy_dataset.with_format.return_value = mock_dummy_dataset # Chain with_format
    mock_hf_load_dataset.return_value = mock_dummy_dataset

    with pytest.raises(NotImplementedError):
        utils.load_data('unsupported_dataset')
    mock_hf_load_dataset.assert_called_once_with('unsupported_dataset', cache_dir="./data")

# --- Tests for Class Distribution ---

def test_compute_class_distribution_correct_counts():
    """Test compute_class_distribution for correct class counts."""
    labels = jnp.array([0, 1, 0, 2, 1, 0, 0, 2, 3])
    num_classes = 4
    expected_counts = jnp.array([4, 2, 2, 1])
    counts = utils.compute_class_distribution(labels, num_classes)
    assert jnp.array_equal(counts, expected_counts)

def test_compute_class_distribution_empty_classes():
    """Test with classes that have no samples."""
    labels = jnp.array([0, 0, 2, 2])
    num_classes = 4 # Class 1 and 3 are empty
    expected_counts = jnp.array([2, 0, 2, 0])
    counts = utils.compute_class_distribution(labels, num_classes)
    assert jnp.array_equal(counts, expected_counts)

# --- Tests for Model Summary ---

def test_print_model_summary_output_structure(key):
    """Test the structure of the output from print_model_summary."""
    # Create a mock model with the necessary structure for summary
    mock_model = init_noprop_model(key, num_classes=10, embed_dim=32, input_channels=1, time_emb_dim_internal=16)

    input_shape = (1, 28, 28)
    
    captured_output = io.StringIO()
    with contextlib.redirect_stdout(captured_output):
        utils.print_model_summary(mock_model, input_shape)
    
    output_str = captured_output.getvalue()
    
    assert "NoProp CT Model Summary" in output_str
    assert f"Embedding dimension: {mock_model.embed_dim}" in output_str
    assert f"Number of classes: {mock_model.num_classes}" in output_str
    assert "CNN parameters:" in output_str
    assert "Total parameters:" in output_str

# --- Tests for Model Evaluation ---

@mock.patch('nullaprop.inference.inference_ct_heun') # Patched at source
@mock.patch('nullaprop.training.compute_loss_aligned') # Patched at source
def test_evaluate_model_metrics_calculation(mock_compute_loss_aligned, mock_inference_ct_heun, key):
    """Test evaluate_model for correct metrics calculation."""
    mock_model_eval = mock.MagicMock() # A simple mock for the model object
    
    # Mock inference and loss outputs
    # Batch 1: 2 correct out of 3
    mock_inference_ct_heun.side_effect = [
        jnp.array([0, 1, 0]), # Predictions for batch 1
        jnp.array([2, 2])     # Predictions for batch 2
    ]
    mock_compute_loss_aligned.side_effect = [
        jnp.array(0.5), # Loss for batch 1
        jnp.array(0.3)  # Loss for batch 2
    ]
        
    # Mock test iterator: yields (images, labels)
    # Images don't matter as inference is mocked
    mock_x_batch1 = jnp.zeros((3, 1, 28, 28)) 
    mock_y_batch1 = jnp.array([0, 0, 0]) # True labels for batch 1 (2 correct)
    mock_x_batch2 = jnp.zeros((2, 1, 28, 28))
    mock_y_batch2 = jnp.array([2, 1]) # True labels for batch 2 (1 correct)

    mock_test_iterator = [
        (mock_x_batch1, mock_y_batch1),
        (mock_x_batch2, mock_y_batch2)
    ]
    
    key_eval, _ = jax.random.split(key)
    accuracy, avg_loss = utils.evaluate_model(mock_model_eval, mock_test_iterator, key_eval, T_steps=10)
    
    assert mock_inference_ct_heun.call_count == 2
    assert mock_compute_loss_aligned.call_count == 2
    
    # Total correct: 2 (batch1) + 1 (batch2) = 3
    # Total samples: 3 (batch1) + 2 (batch2) = 5
    # Expected accuracy: 3 / 5 = 0.6
    assert np.isclose(accuracy, 0.6)
    
    # Expected avg_loss: (0.5 + 0.3) / 2 = 0.4
    assert np.isclose(avg_loss, 0.4)

@mock.patch('nullaprop.inference.inference_ct_heun') # Patched at source
@mock.patch('nullaprop.training.compute_loss_aligned') # Patched at source
def test_evaluate_model_num_batches_limit(mock_compute_loss_aligned, mock_inference_ct_heun, key):
    """Test evaluate_model with num_batches limit."""
    mock_model_eval = mock.MagicMock()
    mock_inference_ct_heun.return_value = jnp.array([0]) # Dummy prediction
    mock_compute_loss_aligned.return_value = jnp.array(0.1)   # Dummy loss

    mock_test_iterator = [
        (jnp.zeros((1,1,28,28)), jnp.array([0])),
        (jnp.zeros((1,1,28,28)), jnp.array([0])),
        (jnp.zeros((1,1,28,28)), jnp.array([0]))
    ]
    key_eval, _ = jax.random.split(key)
    utils.evaluate_model(mock_model_eval, mock_test_iterator, key_eval, num_batches=2)
    
    assert mock_inference_ct_heun.call_count == 2 # Should only process 2 batches
    assert mock_compute_loss_aligned.call_count == 2

# --- Tests for Dataset Information ---

def test_get_dataset_info_mnist():
    """Test get_dataset_info for MNIST."""
    info = utils.get_dataset_info("mnist")
    assert info["num_classes"] == 10
    assert info["input_channels"] == 1
    assert info["input_size"] == (28, 28)

def test_get_dataset_info_cifar10():
    """Test get_dataset_info for CIFAR-10."""
    info = utils.get_dataset_info("cifar10")
    assert info["num_classes"] == 10
    assert info["input_channels"] == 3
    assert info["class_names"][0] == 'airplane'

def test_get_dataset_info_cifar100():
    """Test get_dataset_info for CIFAR-100."""
    info = utils.get_dataset_info("cifar100")
    assert info["num_classes"] == 100
    assert info["input_channels"] == 3

def test_get_dataset_info_unknown():
    """Test get_dataset_info for an unknown dataset."""
    with pytest.raises(ValueError, match="Unknown dataset: unknown_dataset"):
        utils.get_dataset_info("unknown_dataset")

# --- Tests for Prototype Initialization ---

def mock_cnn_part(images_batch):
    """A mock CNN part that returns fixed-size features."""
    batch_size = images_batch.shape[0]
    # Simulate features based on the first pixel value, for predictability
    # This is a very simplistic mock, just to get some varying features.
    # A real test might need more sophisticated feature generation or mocking.
    feature_dim = 8 
    features = jnp.array([jnp.full(feature_dim, img[0,0,0]/255.0) for img in images_batch])
    return features

def mock_dataset_loader_fn_proto(num_classes, samples_per_class_total, feature_dim):
    """
    A mock dataset loader that yields predictable image-label pairs for prototype testing.
    Images are simple so mock_cnn_part can produce predictable features.
    """
    all_images = []
    all_labels = []
    for c in range(num_classes):
        for i in range(samples_per_class_total):
            # Create a simple image where the first pixel value is related to class and sample index
            # This helps in making mock_cnn_part produce somewhat distinct features.
            img_val = (c * 10 + i) % 256 
            img = np.full((1, 28, 28), img_val, dtype=np.uint8) # Assuming 1 channel for simplicity
            all_images.append(img)
            all_labels.append(c)
    
    # Yield in one batch for simplicity in this mock
    # In reality, this would yield multiple batches.
    # The function under test concatenates them anyway.
    def loader():
        yield jnp.array(all_images), jnp.array(all_labels)
    return loader


@mock.patch('nullaprop.utils.partial') # To control the JIT compilation if it causes issues in test
def test_initialize_with_prototypes_jax_basic_logic(mock_partial_jit, key):
    """Test basic logic of prototype initialization."""
    mock_partial_jit.side_effect = lambda func, static_argnums: func # Bypass JIT for easier mocking/testing

    num_classes = 2
    samples_per_class_for_medoid = 3 # Samples to pick for medoid calculation
    total_samples_per_class_in_dataset = 5 # Total samples available per class in mock dataset
    
    # Mock CNN output dimension
    # This should ideally be inferred by the function, but for mocking, we can be explicit
    # Our mock_cnn_part will return features of this dimension.
    cnn_feature_dim = 8 

    # Create a dataset loader function that uses our mock data
    loader_fn = mock_dataset_loader_fn_proto(num_classes, total_samples_per_class_in_dataset, cnn_feature_dim)

    # Call the function
    W_proto = utils.initialize_with_prototypes_jax(
        model_cnn_part=mock_cnn_part,
        dataset_loader_fn=loader_fn,
        num_classes=num_classes,
        key=key,
        samples_per_class=samples_per_class_for_medoid
    )

    assert W_proto.shape == (num_classes, cnn_feature_dim)
    # Further assertions would require knowing the exact medoid calculation for the mock data.
    # For now, checking shape and that it runs without error is a good start.
    # A more detailed test would involve:
    # 1. Manually creating a small set of features for each class.
    # 2. Manually calculating the medoid for those features.
    # 3. Asserting W_proto matches these manually calculated medoids.
    # This is complex due to the randomness in `jax.random.choice`.
    # We can at least check that prototypes are not all zeros if data exists.
    assert not jnp.all(W_proto == 0)


def test_initialize_with_prototypes_jax_no_samples_for_class(key):
    """Test prototype initialization when a class has no samples."""
    num_classes = 2
    cnn_feature_dim = 8

    # Mock dataset loader: class 0 has samples, class 1 has none
    def loader_fn_empty_class():
        # Class 0 images/labels
        imgs_c0 = jnp.array([np.full((1, 28, 28), 10, dtype=np.uint8)] * 3)
        labels_c0 = jnp.array([0] * 3)
        # No samples for class 1
        yield imgs_c0, labels_c0
    
    captured_output = io.StringIO()
    with contextlib.redirect_stdout(captured_output):
        W_proto = utils.initialize_with_prototypes_jax(
            model_cnn_part=mock_cnn_part,
            dataset_loader_fn=loader_fn_empty_class,
            num_classes=num_classes,
            key=key,
            samples_per_class=2
        )
    
    output_str = captured_output.getvalue()
    assert f"Warning: No samples found for class 1" in output_str
    assert W_proto.shape == (num_classes, cnn_feature_dim)
    assert not jnp.all(W_proto[0] == 0) # Class 0 should have a non-zero prototype
    assert jnp.all(W_proto[1] == 0)     # Class 1 should have a zero prototype
