
# Datasets for Numpy

![](mnist_image.png)

## Requirements

- Python 3.x
- Numpy

## Datasets

- [MNIST](http://yann.lecun.com/exdb/mnist/index.html)
- [KMNIST](https://github.com/rois-codh/kmnist)
- [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)
- [Banknote Authentication](https://archive.ics.uci.edu/ml/datasets/banknote+authentication)
- [Sonar (Mines vs. Rocks)](http://archive.ics.uci.edu/ml/datasets/connectionist+bench+(sonar,+mines+vs.+rocks)
- [MHEALTH](https://archive.ics.uci.edu/ml/datasets/MHEALTH+Dataset)

## Usage

First, download `loader.py` from this repository and locate it to your working directory.
You can then load dataset splits into numpy arrays as follows :

```python
from loader import load_dataset

x_train, y_train, x_test, y_test, classes = load_dataset('MNIST')  # either MNIST, Fashion-MNIST, KMNIST, Banknote, Sonar or MHEALTH
```
classes is a tuple of strings representing class names.

The module checks if the relevant .pkl file is already available under ./data. Otherwise, the dataset will be downloaded and processed into a .pkl file.

Labels are integer encoded by default. One-hot encoded labels can be retrieved like so:

```python
from loader import load_dataset

x_train, y_train, x_test, y_test, classes = load_dataset('KMNIST', one_hot=True)
```

You can change the proportion of samples allocated to training set by specifying the train_prop argument when
loading datasets for the first time (other than MNIST, KMNIST and Fashion-MNIST, which have predefined test sets). Default is 80 %.

```python
from loader import load_dataset

x_train, y_train, x_test, y_test, classes = load_dataset('Sonar', train_prop=0.7)
```

Thanks to hsjeong5 for his work on MNIST-for-Numpy!

