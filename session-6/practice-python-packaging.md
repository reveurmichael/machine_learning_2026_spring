# Converting a Neural Network Script into a Python Package

**Goal:** Transform a single neural network script (`code-my_nn.py`) into a professional Python package (`simple_nn_utseus`) that can be installed via pip, published to PyPI (conceptually), and demonstrates proper Python packaging concepts.

**Inspirations**:
- SpaceMining:
    - https://github.com/reveurmichael/space_mining
    - https://pypi.org/project/space-mining/
- scikit-learn:
    - https://github.com/scikit-learn/scikit-learn
    - https://pypi.org/project/scikit-learn/

---

## 1️⃣ Understanding the Original Script

Your `code-my_nn.py` contains several components:

- **Data loading**: `load_mnist_from_csv()`
- **Layer classes**: `Layer`, `Dense`, `ReLU`
- **Loss function**: `softmax_crossentropy_with_logits()`
- **Training utilities**: `forward()`, `predict()`, `train()`
- **Main training function**: `train_mnist_network()`

**Key Challenge**: This is all in one file. We need to organize it into a **modular package structure**.

---

## 2️⃣ Package Structure Design

```
simple_nn_utseus/
├── simple_nn_utseus/              # Main package directory
│   ├── __init__.py                # Package-level API (exports public interface)
│   ├── data/                      # Data loading subpackage
│   │   ├── __init__.py            # Exports data utilities
│   │   ├── mnist_loader.py        # MNIST data loading
│   │   └── _preprocessing.py      # Internal preprocessing (not exported)
│   ├── layers/                    # Neural network layers subpackage
│   │   ├── __init__.py            # Exports layer classes
│   │   ├── base.py                # Base Layer class
│   │   ├── dense.py               # Dense layer
│   │   ├── activation.py          # ReLU and other activations
│   │   └── _utils.py              # Internal utilities (not exported)
│   ├── losses/                    # Loss functions subpackage
│   │   ├── __init__.py            # Exports loss functions
│   │   └── crossentropy.py        # Cross-entropy implementation
│   ├── training/                  # Training utilities subpackage
│   │   ├── __init__.py            # Exports training functions
│   │   ├── trainer.py             # Training loop
│   │   └── _metrics.py            # Internal metrics (not exported)
│   └── utils/                     # General utilities
│       ├── __init__.py            
│       └── helpers.py             # Helper functions
├── tests/                         # Test directory
│   ├── __init__.py
│   ├── test_layers.py
│   └── test_training.py
├── examples/                      # Example usage scripts
│   └── train_mnist_example.py
├── README.md                      # Package documentation
├── LICENSE                        # License file
└── pyproject.toml                 # Modern packaging configuration
```

**Design Principles**:
- **Subpackages** (`data/`, `layers/`, etc.) group related functionality
- **`__init__.py`** files control what is publicly accessible
- Files starting with `_` (like `_utils.py`) are **internal/private** by convention
- Each `__init__.py` explicitly exports only what should be public

---

## 3️⃣ Creating the Package Step-by-Step

### Step 3.1: Create Directory Structure

```bash
mkdir -p simple_nn_utseus/simple_nn_utseus/{data,layers,losses,training,utils}
mkdir -p simple_nn_utseus/{tests,examples}
cd simple_nn_utseus
```

### Step 3.2: Data Subpackage

**`simple_nn_utseus/data/mnist_loader.py`**

```python
"""MNIST data loading utilities."""
import pandas as pd
import numpy as np


def load_mnist_from_csv(train_csv_path, test_csv_path, val_split=0.1):
    """
    Load MNIST dataset from CSV files and split into train/val/test sets.
    
    Args:
        train_csv_path: Path to training CSV file
        test_csv_path: Path to test CSV file
        val_split: Fraction of training data to use for validation
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    train_data = pd.read_csv(train_csv_path)
    test_data = pd.read_csv(test_csv_path)
    
    y_train_full = train_data.iloc[:, 0].to_numpy(np.int64)
    X_train_full = train_data.iloc[:, 1:].to_numpy(np.float32) / 255.0
    
    y_test = test_data.iloc[:, 0].to_numpy(np.int64)
    X_test = test_data.iloc[:, 1:].to_numpy(np.float32) / 255.0
    
    # Split validation set
    n_val = int(len(X_train_full) * val_split)
    np.random.seed(42)
    val_indices = np.random.choice(len(X_train_full), n_val, replace=False)
    
    train_mask = np.ones(len(X_train_full), dtype=bool)
    train_mask[val_indices] = False
    
    X_val = X_train_full[val_indices]
    y_val = y_train_full[val_indices]
    X_train = X_train_full[train_mask]
    y_train = y_train_full[train_mask]
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test
```

**`simple_nn_utseus/data/_preprocessing.py`** (internal module)

```python
"""Internal preprocessing utilities (not exposed to users)."""
import numpy as np


def _normalize_images(images):
    """Internal function to normalize image data."""
    return images / 255.0


def _create_validation_split(X, y, val_split=0.1, seed=42):
    """Internal function to create validation split."""
    n_val = int(len(X) * val_split)
    np.random.seed(seed)
    val_indices = np.random.choice(len(X), n_val, replace=False)
    train_mask = np.ones(len(X), dtype=bool)
    train_mask[val_indices] = False
    return X[train_mask], y[train_mask], X[val_indices], y[val_indices]
```

**`simple_nn_utseus/data/__init__.py`**

```python
"""
Data loading and preprocessing utilities.

This subpackage provides functions for loading datasets like MNIST.
"""

# Export public API
from .mnist_loader import load_mnist_from_csv

# Define what gets imported with "from simple_nn_utseus.data import *"
__all__ = [
    'load_mnist_from_csv',
]

# Note: _preprocessing module is NOT exported (it's internal)
```

### Step 3.3: Layers Subpackage

**`simple_nn_utseus/layers/base.py`**

```python
"""Base layer class."""
import numpy as np


class Layer:
    """Base class for all neural network layers."""
    
    def forward(self, input):
        """
        Forward pass.
        
        Args:
            input: Input data
            
        Returns:
            Output of the layer
        """
        return input
    
    def backward(self, grad_output):
        """
        Backward pass.
        
        Args:
            grad_output: Gradient from the next layer
            
        Returns:
            Gradient with respect to input
        """
        return grad_output
```

**`simple_nn_utseus/layers/dense.py`**

```python
"""Fully connected (Dense) layer implementation."""
import numpy as np
from .base import Layer


class Dense(Layer):
    """Fully connected layer with learnable weights and biases."""
    
    def __init__(self, input_units, output_units, learning_rate=0.1):
        """
        Initialize Dense layer.
        
        Args:
            input_units: Number of input features
            output_units: Number of output features
            learning_rate: Learning rate for gradient descent
        """
        self.learning_rate = learning_rate
        
        # He initialization for better training with ReLU
        self.weights = np.random.randn(input_units, output_units) * np.sqrt(
            2.0 / input_units
        )
        self.biases = np.zeros(output_units)
    
    def forward(self, input):
        """Compute forward pass: output = input @ weights + biases."""
        self.input = input
        return np.dot(input, self.weights) + self.biases
    
    def backward(self, grad_output):
        """
        Compute backward pass and update parameters.
        
        Args:
            grad_output: Gradient of loss with respect to layer output
            
        Returns:
            Gradient of loss with respect to layer input
        """
        # Compute gradients
        grad_weights = np.dot(self.input.T, grad_output)
        grad_biases = np.sum(grad_output, axis=0)
        grad_input = np.dot(grad_output, self.weights.T)
        
        # Update parameters
        self.weights -= self.learning_rate * grad_weights
        self.biases -= self.learning_rate * grad_biases
        
        return grad_input
```

**`simple_nn_utseus/layers/activation.py`**

```python
"""Activation function layers."""
import numpy as np
from .base import Layer


class ReLU(Layer):
    """Rectified Linear Unit activation function."""
    
    def forward(self, input):
        """Apply ReLU: max(0, x)."""
        self.input = input
        return np.maximum(0, input)
    
    def backward(self, grad_output):
        """
        Compute gradient: 1 if x > 0, else 0.
        
        Args:
            grad_output: Gradient from the next layer
            
        Returns:
            Gradient multiplied by ReLU derivative
        """
        relu_grad = self.input > 0
        return grad_output * relu_grad


class Sigmoid(Layer):
    """Sigmoid activation function (example of additional functionality)."""
    
    def forward(self, input):
        """Apply sigmoid: 1 / (1 + exp(-x))."""
        self.output = 1 / (1 + np.exp(-input))
        return self.output
    
    def backward(self, grad_output):
        """Compute gradient: sigmoid(x) * (1 - sigmoid(x))."""
        return grad_output * self.output * (1 - self.output)
```

**`simple_nn_utseus/layers/_utils.py`** (internal)

```python
"""Internal layer utilities (not exposed to users)."""
import numpy as np


def _he_initialization(input_size, output_size):
    """Internal: He initialization for weights."""
    return np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)


def _xavier_initialization(input_size, output_size):
    """Internal: Xavier initialization for weights."""
    return np.random.randn(input_size, output_size) * np.sqrt(1.0 / input_size)
```

**`simple_nn_utseus/layers/__init__.py`**

```python
"""
Neural network layers.

This subpackage provides various layer types for building neural networks.
"""

# Import from submodules
from .base import Layer
from .dense import Dense
from .activation import ReLU, Sigmoid

# Public API
__all__ = [
    'Layer',
    'Dense',
    'ReLU',
    'Sigmoid',
]

# Note: _utils is internal and not exported
```

### Step 3.4: Losses Subpackage

**`simple_nn_utseus/losses/crossentropy.py`**

```python
"""Cross-entropy loss implementations."""
import numpy as np


def softmax_crossentropy_with_logits(logits, labels):
    """
    Compute softmax cross-entropy loss and gradient.
    
    Args:
        logits: Raw network outputs (before softmax), shape (batch_size, num_classes)
        labels: True class labels, shape (batch_size,)
        
    Returns:
        Tuple of (loss, gradient):
            - loss: Scalar loss value
            - gradient: Gradient of loss with respect to logits
    """
    batch_size = logits.shape[0]
    
    # Create one-hot encoded labels
    one_hot_labels = np.zeros_like(logits)
    one_hot_labels[np.arange(batch_size), labels] = 1
    
    # Compute softmax with numerical stability
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # Compute cross-entropy loss
    loss = -np.sum(one_hot_labels * np.log(softmax_probs + 1e-9)) / batch_size
    
    # Gradient of cross-entropy with respect to logits
    grad = (softmax_probs - one_hot_labels) / batch_size
    
    return loss, grad


def softmax(logits):
    """
    Compute softmax probabilities.
    
    Args:
        logits: Raw scores, shape (batch_size, num_classes)
        
    Returns:
        Probability distribution over classes
    """
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
```

**`simple_nn_utseus/losses/__init__.py`**

```python
"""
Loss functions for neural network training.

This subpackage provides various loss functions and their gradients.
"""

from .crossentropy import softmax_crossentropy_with_logits, softmax

__all__ = [
    'softmax_crossentropy_with_logits',
    'softmax',
]
```

### Step 3.5: Training Subpackage

**`simple_nn_utseus/training/trainer.py`**

```python
"""Training utilities for neural networks."""
import numpy as np
from ..losses.crossentropy import softmax_crossentropy_with_logits, softmax


def forward(network, X):
    """
    Perform forward pass through the network.
    
    Args:
        network: List of layer objects
        X: Input data
        
    Returns:
        List of activations from each layer
    """
    activations = []
    input_data = X
    
    for layer in network:
        input_data = layer.forward(input_data)
        activations.append(input_data)
    
    return activations


def predict(network, X):
    """
    Get class predictions from the network.
    
    Args:
        network: List of layer objects
        X: Input data
        
    Returns:
        Predicted class labels
    """
    logits = forward(network, X)[-1]
    probs = softmax(logits)
    return np.argmax(probs, axis=-1)


def train(network, X, y):
    """
    Train the network on a batch of data.
    
    Args:
        network: List of layer objects
        X: Input data
        y: True labels
        
    Returns:
        Loss value
    """
    # Forward pass
    activations = forward(network, X)
    logits = activations[-1]
    
    # Compute loss and gradient
    loss, grad_logits = softmax_crossentropy_with_logits(logits, y)
    
    # Backward pass
    grad_output = grad_logits
    for layer in reversed(network):
        grad_output = layer.backward(grad_output)
    
    return loss


def train_network(network, X_train, y_train, X_val, y_val, num_epochs=200):
    """
    Full training loop for a neural network.
    
    Args:
        network: List of layer objects
        X_train: Training data
        y_train: Training labels
        X_val: Validation data
        y_val: Validation labels
        num_epochs: Number of training epochs
    """
    print(f"Training on {len(X_train)} examples using Gradient Descent")
    
    for epoch in range(num_epochs):
        # Train on entire dataset (full batch GD)
        loss = train(network, X_train, y_train)
        
        # Compute training accuracy
        train_predictions = predict(network, X_train)
        train_accuracy = np.mean(train_predictions == y_train)
        
        # Compute validation accuracy
        val_predictions = predict(network, X_val)
        val_accuracy = np.mean(val_predictions == y_val)
        
        print(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Loss: {loss:.4f}, "
            f"Train Acc: {train_accuracy:.4f}, "
            f"Val Acc: {val_accuracy:.4f}"
        )
```

**`simple_nn_utseus/training/_metrics.py`** (internal)

```python
"""Internal metrics computation (not exposed to users)."""
import numpy as np


def _compute_accuracy(predictions, labels):
    """Internal: Compute classification accuracy."""
    return np.mean(predictions == labels)


def _compute_confusion_matrix(predictions, labels, num_classes):
    """Internal: Compute confusion matrix."""
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(labels, predictions):
        matrix[true, pred] += 1
    return matrix
```

**`simple_nn_utseus/training/__init__.py`**

```python
"""
Training utilities and loops.

This subpackage provides functions for training neural networks.
"""

from .trainer import forward, predict, train, train_network

__all__ = [
    'forward',
    'predict',
    'train',
    'train_network',
]

# _metrics is internal and not exported
```

### Step 3.6: Utils Subpackage

**`simple_nn_utseus/utils/helpers.py`**

```python
"""General utility functions."""
import numpy as np


def set_random_seed(seed=42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)


def print_network_architecture(network):
    """
    Print network architecture summary.
    
    Args:
        network: List of layer objects
    """
    print("Network Architecture:")
    print("-" * 50)
    for i, layer in enumerate(network):
        layer_type = layer.__class__.__name__
        if hasattr(layer, 'weights'):
            shape = f"{layer.weights.shape[0]} → {layer.weights.shape[1]}"
            print(f"Layer {i}: {layer_type:15} ({shape})")
        else:
            print(f"Layer {i}: {layer_type:15}")
    print("-" * 50)
```

**`simple_nn_utseus/utils/__init__.py`**

```python
"""
General utility functions.

This subpackage provides miscellaneous helper functions.
"""

from .helpers import set_random_seed, print_network_architecture

__all__ = [
    'set_random_seed',
    'print_network_architecture',
]
```

### Step 3.7: Main Package `__init__.py`

**`simple_nn_utseus/__init__.py`**

```python
"""
Simple Neural Network Package (simple_nn_utseus)

A simple, educational neural network library implemented from scratch in NumPy.

Main Components:
    - data: Data loading utilities
    - layers: Neural network layer implementations
    - losses: Loss functions
    - training: Training utilities
    - utils: General helper functions

Example:
    >>> from simple_nn_utseus import Dense, ReLU, load_mnist_from_csv, train_network
    >>> X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_from_csv(
    ...     "mnist_train.csv", "mnist_test.csv"
    ... )
    >>> network = [Dense(784, 64), ReLU(), Dense(64, 10)]
    >>> train_network(network, X_train, y_train, X_val, y_val, num_epochs=10)
"""

__version__ = "0.1.0"
__author__ = "Your Name"

# Import from subpackages - expose the most commonly used items
from .data import load_mnist_from_csv
from .layers import Layer, Dense, ReLU, Sigmoid
from .losses import softmax_crossentropy_with_logits, softmax
from .training import forward, predict, train, train_network
from .utils import set_random_seed, print_network_architecture

# Define public API
__all__ = [
    # Data utilities
    'load_mnist_from_csv',
    
    # Layers
    'Layer',
    'Dense',
    'ReLU',
    'Sigmoid',
    
    # Losses
    'softmax_crossentropy_with_logits',
    'softmax',
    
    # Training
    'forward',
    'predict',
    'train',
    'train_network',
    
    # Utils
    'set_random_seed',
    'print_network_architecture',
]
```

---

## 4️⃣ Configuration with pyproject.toml

**`pyproject.toml`** is the **modern standard** (PEP 517, 518, 621) for Python packaging. It replaces `setup.py` and `setup.cfg`.

**`pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "simple_nn_utseus"
version = "0.1.0"
description = "A simple neural network library for educational purposes"
readme = "README.md"
requires-python = ">=3.8"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
maintainers = [
    {name = "Your Name", email = "your.email@example.com"}
]
keywords = ["neural-network", "machine-learning", "education", "numpy"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Education",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
dependencies = [
    "numpy>=1.20.0",
    "pandas>=1.3.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=3.0",
    "black>=22.0",
    "flake8>=4.0",
]

[project.urls]
Homepage = "https://github.com/yourusername/simple_nn_utseus"
Documentation = "https://github.com/yourusername/simple_nn_utseus#readme"
Repository = "https://github.com/yourusername/simple_nn_utseus.git"
"Bug Tracker" = "https://github.com/yourusername/simple_nn_utseus/issues"

[tool.setuptools]
packages = ["simple_nn_utseus", "simple_nn_utseus.data", "simple_nn_utseus.layers", 
            "simple_nn_utseus.losses", "simple_nn_utseus.training", "simple_nn_utseus.utils"]

[tool.setuptools.package-data]
simple_nn_utseus = ["py.typed"]
```

**Key Sections Explained**:

- **`[build-system]`**: Specifies how to build the package (uses setuptools)
- **`[project]`**: Package metadata (name, version, dependencies)
- **`[project.optional-dependencies]`**: Optional dependencies (e.g., for development)
- **`[project.urls]`**: Links to documentation, repository, etc.
- **`[tool.setuptools]`**: Setuptools-specific configuration

---

## 5️⃣ README.md

See [README.md](./README.md).

---

## 6️⃣ Local Installation

### Install in Editable Mode

```bash
cd simple_nn_utseus
pip install -e .
```

**What does `-e` (editable mode) do?**
- Changes to your source code are immediately reflected without reinstalling
- Perfect for development
- Creates a link to your source directory instead of copying files

### Verify Installation

```python
import simple_nn_utseus
print(simple_nn_utseus.__version__)  # Should print 0.1.0

# Test imports
from simple_nn_utseus import Dense, ReLU, load_mnist_from_csv
print("Successfully imported!")
```

---

## 7️⃣ Understanding Imports

### What `__init__.py` Does

**`__init__.py`** serves two main purposes:

1. **Marks a directory as a Python package**
2. **Controls what gets exported** when someone imports your package

### Import Examples

```python
# 1. Import from top-level package (most common)
from simple_nn_utseus import Dense, ReLU, load_mnist_from_csv

# 2. Import subpackages
from simple_nn_utseus.layers import Dense, ReLU
from simple_nn_utseus.data import load_mnist_from_csv

# 3. Import specific modules
from simple_nn_utseus.layers.dense import Dense
from simple_nn_utseus.layers.activation import ReLU

# 4. Import entire subpackage
import simple_nn_utseus.layers as layers

network = [layers.Dense(784, 64), layers.ReLU()]

# 5. What you should NOT do
from simple_nn_utseus.layers import _utils  # Not in __all__
from simple_nn_utseus.data import _preprocessing  # Not exported
```

### Understanding `__all__`

```python
# In simple_nn_utseus/layers/__init__.py
__all__ = ['Layer', 'Dense', 'ReLU', 'Sigmoid']

# This controls what gets imported with:
from simple_nn_utseus.layers import *  # Only imports Layer, Dense, ReLU, Sigmoid
```

---

## 8️⃣ Publishing to PyPI (Overview)

**Important Note**: Publishing to PyPI makes your package available to everyone via `pip install`. For educational/practice packages like this, **you don't need to publish** unless it's truly useful to others. However, understanding the process is valuable.

### The Publishing Process (High-Level)

1. **Create PyPI Account**
   - Go to https://pypi.org and register
   - Verify your email

2. **Install Publishing Tools**
   ```bash
   pip install build twine
   ```
   
3. **Build Your Package**
   ```bash
   # In your package root directory (where pyproject.toml is)
   python -m build
   ```
   
   This creates two files in the `dist/` directory:
   - `simple_nn_utseus-0.1.0.tar.gz` (source distribution)
   - `simple_nn_utseus-0.1.0-py3-none-any.whl` (wheel distribution)

4. **Test on TestPyPI First** (Recommended)
   ```bash
   # Upload to TestPyPI (a separate testing instance)
   python -m twine upload --repository testpypi dist/*
   
   # Try installing from TestPyPI
   pip install --index-url https://test.pypi.org/simple/ simple_nn_utseus
   ```

5. **Upload to Real PyPI**
   ```bash
   python -m twine upload dist/*
   ```
   
   Enter your PyPI username and password when prompted.

6. **Install Your Package**
   ```bash
   pip install simple_nn_utseus
   ```

### Why You Probably Shouldn't Publish This

- **PyPI is for production-ready packages**: This is an educational project
- **Name squatting**: Don't claim package names you won't maintain
- **Responsibility**: Published packages should be maintained and documented
- **Namespace pollution**: Too many low-quality packages make PyPI harder to use

### When You SHOULD Publish

- Your package solves a real problem
- It's well-tested and documented
- You're committed to maintaining it
- It adds value that existing packages don't provide

### Alternative: Private Package Repository

For practice or internal use, consider:
- **GitHub/GitLab**: Install directly via `pip install git+https://github.com/user/repo.git`
- **Private PyPI server**: Tools like `devpi` or `pypiserver`
- **Just use local installation**: `pip install -e .` works great for development

---

## 9️⃣ Example Usage Script

Create **`examples/train_mnist_example.py`**:

```python
"""
Example: Training a neural network on MNIST dataset.

This script demonstrates how to use the simple_nn_utseus package
to build and train a neural network on the MNIST dataset.
"""

from simple_nn_utseus import (
    load_mnist_from_csv,
    Dense,
    ReLU,
    train_network,
    print_network_architecture,
    set_random_seed
)


def main():
    """Main training function."""
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # Load MNIST data
    print("Loading MNIST data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_from_csv(
        "mnist_train.csv",
        "mnist_test.csv",
        val_split=0.1
    )
    
    # Build network architecture
    network = [
        Dense(784, 128, learning_rate=0.1),  # Input: 784 (28x28 images)
        ReLU(),
        Dense(128, 64, learning_rate=0.1),
        ReLU(),
        Dense(64, 10, learning_rate=0.1)     # Output: 10 classes (0-9)
    ]
    
    # Print network structure
    print_network_architecture(network)
    
    # Train the network
    print("\nStarting training...")
    train_network(
        network=network,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        num_epochs=50
    )
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
```

**Run the example:**
```bash
python examples/train_mnist_example.py
```

---

## 🔟 Testing Your Package

### Step 10.1: Create Test Files

**`tests/__init__.py`**
```python
"""Tests for simple_nn_utseus package."""
```

**`tests/test_layers.py`**
```python
"""Tests for neural network layers."""
import numpy as np
import pytest
from simple_nn_utseus.layers import Dense, ReLU, Layer


class TestLayer:
    """Test base Layer class."""
    
    def test_forward(self):
        """Test that base layer returns input unchanged."""
        layer = Layer()
        x = np.array([[1, 2, 3]])
        output = layer.forward(x)
        np.testing.assert_array_equal(output, x)
    
    def test_backward(self):
        """Test that base layer returns gradient unchanged."""
        layer = Layer()
        grad = np.array([[1, 2, 3]])
        output = layer.backward(grad)
        np.testing.assert_array_equal(output, grad)


class TestDense:
    """Test Dense layer."""
    
    def test_initialization(self):
        """Test Dense layer initialization."""
        layer = Dense(10, 5, learning_rate=0.01)
        assert layer.weights.shape == (10, 5)
        assert layer.biases.shape == (5,)
        assert layer.learning_rate == 0.01
    
    def test_forward_shape(self):
        """Test forward pass output shape."""
        layer = Dense(10, 5)
        x = np.random.randn(32, 10)  # Batch of 32 samples
        output = layer.forward(x)
        assert output.shape == (32, 5)
    
    def test_backward_updates_weights(self):
        """Test that backward pass updates weights."""
        layer = Dense(10, 5, learning_rate=0.1)
        x = np.random.randn(32, 10)
        
        # Forward pass
        output = layer.forward(x)
        
        # Save original weights
        original_weights = layer.weights.copy()
        original_biases = layer.biases.copy()
        
        # Backward pass with random gradient
        grad = np.random.randn(32, 5)
        layer.backward(grad)
        
        # Check that weights changed
        assert not np.allclose(layer.weights, original_weights)
        assert not np.allclose(layer.biases, original_biases)


class TestReLU:
    """Test ReLU activation."""
    
    def test_forward_positive(self):
        """Test ReLU with positive inputs."""
        layer = ReLU()
        x = np.array([[1, 2, 3]])
        output = layer.forward(x)
        np.testing.assert_array_equal(output, x)
    
    def test_forward_negative(self):
        """Test ReLU with negative inputs."""
        layer = ReLU()
        x = np.array([[-1, -2, -3]])
        output = layer.forward(x)
        np.testing.assert_array_equal(output, np.array([[0, 0, 0]]))
    
    def test_forward_mixed(self):
        """Test ReLU with mixed inputs."""
        layer = ReLU()
        x = np.array([[-1, 2, -3, 4]])
        output = layer.forward(x)
        np.testing.assert_array_equal(output, np.array([[0, 2, 0, 4]]))
    
    def test_backward(self):
        """Test ReLU backward pass."""
        layer = ReLU()
        x = np.array([[-1, 2, -3, 4]])
        layer.forward(x)
        
        grad = np.array([[1, 1, 1, 1]])
        output_grad = layer.backward(grad)
        
        # Gradient should be 0 where input was negative, 1 where positive
        expected = np.array([[0, 1, 0, 1]])
        np.testing.assert_array_equal(output_grad, expected)
```

**`tests/test_training.py`**
```python
"""Tests for training utilities."""
import numpy as np
from simple_nn_utseus.layers import Dense, ReLU
from simple_nn_utseus.training import forward, predict, train


class TestTraining:
    """Test training utilities."""
    
    def test_forward_pass(self):
        """Test forward pass through network."""
        network = [
            Dense(10, 5),
            ReLU(),
            Dense(5, 3)
        ]
        x = np.random.randn(32, 10)
        activations = forward(network, x)
        
        # Should have one activation per layer
        assert len(activations) == 3
        # Final output should have correct shape
        assert activations[-1].shape == (32, 3)
    
    def test_predict(self):
        """Test prediction function."""
        network = [
            Dense(10, 5),
            ReLU(),
            Dense(5, 3)
        ]
        x = np.random.randn(32, 10)
        predictions = predict(network, x)
        
        # Should return class indices
        assert predictions.shape == (32,)
        assert predictions.dtype in [np.int32, np.int64]
        # All predictions should be valid class indices (0, 1, or 2)
        assert all(0 <= p < 3 for p in predictions)
    
    def test_train_reduces_loss(self):
        """Test that training reduces loss over time."""
        # Simple problem: learn identity-like mapping
        np.random.seed(42)
        network = [Dense(10, 10, learning_rate=0.1)]
        
        # Generate simple data
        X = np.eye(10)  # Identity matrix
        y = np.arange(10)  # Labels 0-9
        
        # Train for several iterations
        losses = []
        for _ in range(100):
            loss = train(network, X, y)
            losses.append(loss)
        
        # Loss should generally decrease (check first vs last)
        assert losses[-1] < losses[0], "Loss should decrease during training"
```

**`tests/test_data.py`**
```python
"""Tests for data loading."""

import numpy as np
import pandas as pd
import tempfile
import os
from simple_nn_utseus.data import load_mnist_from_csv


class TestDataLoading:
    """Test data loading functions."""

    def test_load_mnist_from_csv(self):
        """Test MNIST CSV loading with synthetic data."""
        # Create temporary CSV files with synthetic data
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = os.path.join(tmpdir, "train.csv")
            test_path = os.path.join(tmpdir, "test.csv")

            # Create synthetic training data (100 samples, 784 features)
            train_data = np.random.randint(0, 255, (100, 785))
            train_data[:, 0] = np.random.randint(0, 10, 100)  # Labels
            pd.DataFrame(train_data).to_csv(train_path, index=False, header=False)

            # Create synthetic test data (20 samples)
            test_data = np.random.randint(0, 255, (20, 785))
            test_data[:, 0] = np.random.randint(0, 10, 20)  # Labels
            pd.DataFrame(test_data).to_csv(test_path, index=False, header=False)

            # Load data
            X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_from_csv(
                train_path, test_path, val_split=0.20000
            )
            
            assert X_train.shape[1] == 784  # 784 features

            # Check data range (should be normalized to [0, 1])
            assert X_train.min() >= 0
            assert X_train.max() <= 1

            # Check label range
            assert y_train.min() >= 0
            assert y_train.max() <= 9
```

### Step 10.2: Run Tests

```bash
# Install pytest if not already installed
pip install pytest pytest-cov

# Run all tests
pytest tests/

# Run with coverage report
pytest tests/ --cov=simple_nn_utseus --cov-report=html

# Run specific test file
pytest tests/test_layers.py

# Run with verbose output
pytest tests/ -v
```

---

## 1️⃣1️⃣ Key Concepts Summary

### **Module vs Package vs Subpackage**

| Concept | Definition | Example |
|---------|------------|---------|
| **Module** | Any `.py` file | `dense.py`, `activation.py` |
| **Package** | Directory with `__init__.py` | `simple_nn_utseus/` |
| **Subpackage** | Package inside another package | `simple_nn_utseus/layers/` |

### **Public vs Private (Internal) Components**

```python
# Public (exported in __init__.py)
from simple_nn_utseus import Dense  # ✅ Works

# Private (not in __all__, starts with _)
from simple_nn_utseus.layers import _utils  # ❌ Not intended for users
```

**Convention**:
- **Public**: Listed in `__all__`, documented for users
- **Private**: Prefixed with `_`, for internal use only

### **Import Hierarchy**

```python
# Level 1: Top-level package (recommended for users)
from simple_nn_utseus import Dense, ReLU

# Level 2: Subpackage
from simple_nn_utseus.layers import Dense, ReLU

# Level 3: Specific module
from simple_nn_utseus.layers.dense import Dense

# Level 4: Relative imports (inside package code only)
from ..layers import Dense  # From within package
```

### **How `__init__.py` Controls Exports**

**`simple_nn_utseus/layers/__init__.py`**:
```python
from .dense import Dense
from .activation import ReLU, Sigmoid
from .base import Layer

__all__ = ['Layer', 'Dense', 'ReLU', 'Sigmoid']
# _utils is NOT exported - users shouldn't use it directly
```

**Result**:
```python
# This works (explicitly imported)
from simple_nn_utseus.layers import Dense  # ✅

# This works too (in __all__)
from simple_nn_utseus.layers import *  # Imports only Dense, ReLU, Sigmoid, Layer

# This doesn't work (not in __all__)
from simple_nn_utseus.layers import _utils  # ❌ (but technically possible)
```

---

## 1️⃣2️⃣ Complete File Checklist

Before finalizing, ensure you have:

- [ ] **Package structure** with all subpackages
- [ ] **All `__init__.py` files** with proper exports
- [ ] **`pyproject.toml`** with complete metadata
- [ ] **`README.md`** with usage examples
- [ ] **`LICENSE`** file (e.g., MIT)
- [ ] **Test files** in `tests/` directory
- [ ] **Example scripts** in `examples/` directory
- [ ] **`.gitignore`** for Python projects

### Sample `.gitignore`

Create **`.gitignore`**:
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
ENV/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Testing
.pytest_cache/
.coverage
htmlcov/

# OS
.DS_Store
Thumbs.db
```

### Sample `LICENSE`

Create **`LICENSE`** (MIT License example):
```
MIT License

Copyright (c) 2100-2200 Your Name
```

---

## 1️⃣3️⃣ Advanced: Adding Type Hints (Optional/Conceptual)

To make your package more professional, add type hints:

**`simple_nn_utseus/layers/dense.py` with type hints**:
```python
"""Fully connected (Dense) layer implementation."""
from typing import Any
import numpy as np
import numpy.typing as npt
from .base import Layer


class Dense(Layer):
    """Fully connected layer with learnable weights and biases."""
    
    def __init__(
        self,
        input_units: int,
        output_units: int,
        learning_rate: float = 0.1
    ) -> None:
        """
        Initialize Dense layer.
        
        Args:
            input_units: Number of input features
            output_units: Number of output features
            learning_rate: Learning rate for gradient descent
        """
        self.learning_rate: float = learning_rate
        
        self.weights: npt.NDArray[np.float64] = np.random.randn(
            input_units, output_units
        ) * np.sqrt(2.0 / input_units)
        self.biases: npt.NDArray[np.float64] = np.zeros(output_units)
        self.input: npt.NDArray[np.float64]
    
    def forward(self, input: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute forward pass: output = input @ weights + biases."""
        self.input = input
        return np.dot(input, self.weights) + self.biases
    
    def backward(
        self, grad_output: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Compute backward pass and update parameters.
        
        Args:
            grad_output: Gradient of loss with respect to layer output
            
        Returns:
            Gradient of loss with respect to layer input
        """
        grad_weights = np.dot(self.input.T, grad_output)
        grad_biases = np.sum(grad_output, axis=0)
        grad_input = np.dot(grad_output, self.weights.T)
        
        self.weights -= self.learning_rate * grad_weights
        self.biases -= self.learning_rate * grad_biases
        
        return grad_input
```

Add to `pyproject.toml`:
```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=3.0",
    "black>=22.0",
    "flake8>=4.0",
    "mypy>=1.0",  # Type checker
]
```

Create **`simple_nn_utseus/py.typed`** (empty file to indicate type hints are available).

---

## 1️⃣4️⃣ Final Package Directory Structure

```
simple_nn_utseus/
├── simple_nn_utseus/                    # Main package
│   ├── __init__.py                      # ✅ Exports main API
│   ├── py.typed                         # Type hints marker (if to be done)
│   │
│   ├── data/                            # Data subpackage
│   │   ├── __init__.py                  # ✅ Exports: load_mnist_from_csv
│   │   ├── mnist_loader.py              # Public module
│   │   └── _preprocessing.py            # Private module (not exported)
│   │
│   ├── layers/                          # Layers subpackage
│   │   ├── __init__.py                  # ✅ Exports: Layer, Dense, ReLU, Sigmoid
│   │   ├── base.py                      # Public: Base Layer class
│   │   ├── dense.py                     # Public: Dense layer
│   │   ├── activation.py                # Public: Activation functions
│   │   └── _utils.py                    # Private: Internal utilities
│   │
│   ├── losses/                          # Losses subpackage
│   │   ├── __init__.py                  # ✅ Exports: softmax_crossentropy_with_logits, softmax
│   │   └── crossentropy.py              # Public: Loss functions
│   │
│   ├── training/                        # Training subpackage
│   │   ├── __init__.py                  # ✅ Exports: forward, predict, train, train_network
│   │   ├── trainer.py                   # Public: Training utilities
│   │   └── _metrics.py                  # Private: Internal metrics
│   │
│   └── utils/                           # Utils subpackage
│       ├── __init__.py                  # ✅ Exports: set_random_seed, print_network_architecture
│       └── helpers.py                   # Public: Helper functions
│
├── tests/                               # Test directory
│   ├── __init__.py                      # ✅ Test package marker
│   ├── test_layers.py                   # Layer tests
│   ├── test_training.py                 # Training tests
│   └── test_data.py                     # Data loading tests
│
├── examples/                            # Example scripts
│   └── train_mnist_example.py           # Usage example
│
├── README.md                            # ✅ Documentation
├── LICENSE                              # ✅ License file
├── pyproject.toml                       # ✅ Package configuration
└── .gitignore                           # ✅ Git ignore rules
```

**Legend**:
- ✅ = Essential file
- Public modules = Exported in `__init__.py`
- Private modules = Start with `_`, not exported

---

## 1️⃣5️⃣ Summary of Learning Objectives

By completing this tutorial, you have learned:

1. ✅ **Python packaging fundamentals**
   - Module, package, subpackage concepts
   - Role of `__init__.py`
   - Public vs private API design

2. ✅ **Modern packaging with `pyproject.toml`**
   - PEP 517/518/621 standards
   - Dependency management
   - Metadata specification

3. ✅ **Code organization**
   - Modular design principles
   - Separation of concerns
   - Clean import hierarchies

4. ✅ **Development workflow**
   - Editable installation (`pip install -e .`)
   - Testing with pytest
   - Documentation with README

5. ✅ **Publishing awareness**
   - How PyPI works (without actually publishing)
   - Build and distribution process
   - Responsibility of package maintainers
