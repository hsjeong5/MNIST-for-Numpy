"""Dataset loader.

Currently supports MNIST, Fashion-MNIST, KMNIST and Banknote.

Forked from hsjeong5.
"""
import numpy as np
from urllib import request
import gzip
import pickle
import os

# Split names, will be used as dictionary keys
SPLITS = ('train_x', 'test_x', 'train_y', 'test_y')

# Files to fetch for MNIST and MNIST-like datasets
MNIST_URL = [
    ["training_images", "train-images-idx3-ubyte.gz"],
    ["test_images", "t10k-images-idx3-ubyte.gz"],
    ["training_labels", "train-labels-idx1-ubyte.gz"],
    ["test_labels", "t10k-labels-idx1-ubyte.gz"]
]

# Links are up-to-date as of August, 27th 2019
URL = {
    'MNIST': 'http://yann.lecun.com/exdb/mnist/',
    'Fashion-MNIST': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws'
                     '.com/',
    'KMNIST': 'http://codh.rois.ac.jp/kmnist/dataset/kmnist/',
    'Banknote': 'http://archive.ics.uci.edu/ml/machine-learning-databases'
                '/00267/data_banknote_authentication.txt'
}

# Class names. Order follows integer encoding in datasets
CLASSES = {
    'MNIST': ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9"),
    'Fashion-MNIST': ("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                      "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"),
    'KMNIST': ('お', 'き', 'す', 'つ', 'な', 'は', 'ま', 'や', 'れ', 'を'),
    'Banknote': ('Genuine', 'Forged')
}


def download_mnist(dataset_name):
    """Download MNIST and MNIST-like datasets."""
    base_url = URL[dataset_name]
    print("Downloading {}...".format(dataset_name))
    for name in MNIST_URL:
        request.urlretrieve(base_url + name[1], name[1])
    print("Download complete.")


def save_mnist(dataset_name):
    """Save dataset as a .pkl file and remove .gz files."""
    data = []
    for name in MNIST_URL[:2]:
        with gzip.open(name[1], 'rb') as f:
            data.append(np.frombuffer(f.read(), np.uint8,
                                      offset=16).reshape(-1, 28 * 28))
    for name in MNIST_URL[-2:]:
        with gzip.open(name[1], 'rb') as f:
            data.append(np.frombuffer(f.read(), np.uint8, offset=8))

    if not os.path.exists('./data'):
        os.mkdir('./data')

    mnist = dict(zip(SPLITS, data))

    with open("data/{}.pkl".format(dataset_name), 'wb') as f:
        pickle.dump(mnist, f)

    for name in MNIST_URL:
        os.remove(name[1])

    print("Save complete.")


def init_mnist(dataset_name):
    download_mnist(dataset_name)
    save_mnist(dataset_name)


def init(dataset_name):
    if not os.path.exists('./data'):
        os.mkdir('./data')

    print("Downloading {}...".format(dataset_name))
    data = request.urlopen(URL[dataset_name])

    data = np.genfromtxt(data, delimiter=',')
    np.random.shuffle(data)

    x, y = data[:, :-1], data[:, -1]
    split = int(0.8 * x.shape[0])  # 80 % to train split

    train_x, train_y = x[0:split], y[0:split]
    test_x, test_y = x[split:-1], y[split:-1]
    train_y = train_y.astype(int)
    test_y = test_y.astype(int)

    dataset = dict(zip(SPLITS, (train_x, test_x, train_y, test_y)))

    with open("data/{}.pkl".format(dataset_name), 'wb') as f:
        pickle.dump(dataset, f)

    print("Save complete.")


def vectorize(labels, dataset_name):
    """Take an integer encoded array and return a one-hot encoded version."""
    temp = np.zeros((len(labels), len(CLASSES[dataset_name])))
    temp[np.arange(len(labels)), labels] = 1
    return temp


def load_dataset(dataset_name='MNIST', one_hot=False):
    """Load dataset as numpy arrays.

    Download dataset if not already available under ./data.
    """
    if dataset_name in ['MNIST', 'KMNIST', 'Fashion-MNIST']:
        if not os.path.exists('data/{}.pkl'.format(dataset_name)):
            init_mnist(dataset_name)

    elif dataset_name in ['Banknote']:
        if not os.path.exists('data/{}.pkl'.format(dataset_name)):
            init(dataset_name)
    else:
        raise Exception('Dataset is not supported.')

    # Load data
    with open("data/{}.pkl".format(dataset_name), 'rb') as f:
        dataset = pickle.load(f)

    if one_hot:
        dataset['train_y'] = vectorize(dataset['train_y'], dataset_name)
        dataset['test_y'] = vectorize(dataset['test_y'], dataset_name)

    return dataset["train_x"], dataset["train_y"], dataset[
        "test_x"], dataset["test_y"], CLASSES[dataset_name]


if __name__ == '__main__':
    init_mnist('MNIST')
