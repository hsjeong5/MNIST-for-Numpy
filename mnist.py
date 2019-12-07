"""MNIST, Fashion-MNIST and KMNIST loader.

Forked from hsjeong5.
"""
import numpy as np
from urllib import request
import gzip
import pickle
import os

filename = [
    ["training_images", "train-images-idx3-ubyte.gz"],
    ["test_images", "t10k-images-idx3-ubyte.gz"],
    ["training_labels", "train-labels-idx1-ubyte.gz"],
    ["test_labels", "t10k-labels-idx1-ubyte.gz"]
]

# Links are up-to-date as of August, 27th 2019
url = {
    'MNIST': 'http://yann.lecun.com/exdb/mnist/',
    'Fashion-MNIST': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws'
                     '.com/',
    'KMNIST': 'http://codh.rois.ac.jp/kmnist/dataset/kmnist/'
}

classes = {
    'MNIST': ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9"),
    'Fashion-MNIST': ("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                      "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"),
    'KMNIST': ('お', 'き', 'す', 'つ', 'な', 'は', 'ま', 'や', 'れ', 'を')
}


def download_mnist(dataset_name):
    """Download dataset."""
    base_url = url[dataset_name]
    print("Downloading {}...".format(dataset_name))
    for name in filename:
        request.urlretrieve(base_url + name[1], name[1])
    print("Download complete.")


def save_mnist(dataset_name):
    """Save dataset as a .pkl file and remove .gz files."""
    mnist = {}
    for name in filename[:2]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8,
                                           offset=16).reshape(-1, 28 * 28)
    for name in filename[-2:]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)

    if not os.path.exists('./data'):
        os.mkdir('./data')

    with open("data/{}.pkl".format(dataset_name), 'wb') as f:
        pickle.dump(mnist, f)

    for name in filename:
        os.remove(name[1])

    print("Save complete.")


def init(dataset_name):
    download_mnist(dataset_name)
    save_mnist(dataset_name)


def vectorize(labels):
    """Take an integer encoded array and return a one-hot encoded version."""
    temp = np.zeros((len(labels), 10))
    temp[np.arange(len(labels)), labels] = 1
    return temp


def load_dataset(dataset_name='MNIST', one_hot=False):
    """Load dataset as numpy arrays.

    Download dataset if not already available under ./data.
    """
    if not os.path.exists('data/{}.pkl'.format(dataset_name)):
        init(dataset_name)

    with open("data/{}.pkl".format(dataset_name), 'rb') as f:
        mnist = pickle.load(f)

    if one_hot:
        mnist['training_labels'] = vectorize(mnist['training_labels'])
        mnist['test_labels'] = vectorize(mnist['test_labels'])

    return mnist["training_images"], mnist["training_labels"], mnist[
        "test_images"], mnist["test_labels"], classes[dataset_name]


if __name__ == '__main__':
    init('MNIST')
