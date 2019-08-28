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


def download_mnist(dataset_name):
    base_url = url[dataset_name]
    print("Downloading {}...".format(dataset_name))
    for name in filename:
        request.urlretrieve(base_url + name[1], name[1])
    print("Download complete.")


def save_mnist(dataset_name):
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


def load(dataset_name='MNIST'):
    if not os.path.exists('data/{}.pkl'.format(dataset_name)):
        init(dataset_name)

    with open("data/{}.pkl".format(dataset_name), 'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist[
        "test_images"], mnist["test_labels"]


if __name__ == '__main__':
    init('MNIST')
