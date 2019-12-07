
# MNIST, KMNIST and Fashion-MNIST for Numpy

![](mnist_image.png)

The MNIST, KMNIST and Fashion-MNIST datasets have 60,000 training examples, and 10,000 test examples each.
Each example included in the datasets is a 28x28 grayscale image and its corresponding label(0-9).
This Python module makes it easy to load the MNIST, KMNIST and Fashion-MNIST datasets into numpy arrays.

For more details about the MNIST dataset, please visit [this link](http://yann.lecun.com/exdb/mnist/index.html).

## Requirements

- Python 3.x
- Numpy

## Usage

First, download `mnist.py` from this repository and locate it to your working directory.
You can then load a dataset as follow :

```python
from mnist import load_dataset

x_train, y_train, x_test, y_test, classes = load_dataset('MNIST')  # either MNIST, Fashion-MNIST or KMNIST
```

The module checks if the relevant .pkl file is already available under ./data. Otherwise, the dataset will be downloaded and processed into a .pkl file.

**load()** takes a .pkl file and returns 5 numpy arrays.

- x_train : 60,000x784 numpy array where each row is a flattened version of of a train image.
- y_train : 60,000x1 numpy array with integer encoded labels.
- x_test : 10,000x784 numpy array where each row is a flattened version of of a test image.
- y_test : 10,000x1 numpy array with integer encoded labels.
- classes : 10x1 numpy array holding class labels.

One-hot encoded labels can be retrieved like so:

```python
from mnist import load_dataset

x_train, y_train, x_test, y_test, classes = load_dataset('KMNIST', one_hot=True)
```

