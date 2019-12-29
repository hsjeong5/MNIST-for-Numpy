"""Dataset loader.

Currently supports MNIST, Fashion-MNIST, KMNIST, Banknote, Sonar and MHEALTH
datasets.

Forked from hsjeong5 for his work on downloading the MNIST dataset.
"""
import numpy as np
from urllib import request
import gzip
import pickle
import os
import zipfile
import shutil

SUPPORTED = ['MNIST', 'KMNIST', 'Fashion-MNIST',
             'Banknote', 'Sonar', 'MHEALTH']

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
                '/00267/data_banknote_authentication.txt',
    'Sonar': 'http://archive.ics.uci.edu/ml/machine-learning-databases'
             '/undocumented/connectionist-bench/sonar/sonar.all-data',
    'MHEALTH': 'https://archive.ics.uci.edu/ml/machine-learning-databases'
               '/00319/MHEALTHDATASET.zip'
}

# Class names. Order follows integer encoding in datasets
CLASSES = {
    'MNIST': ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9"),
    'Fashion-MNIST': ("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                      "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"),
    'KMNIST': ('お', 'き', 'す', 'つ', 'な', 'は', 'ま', 'や', 'れ', 'を'),
    'Banknote': ('Genuine', 'Forged'),
    'Sonar': ('Mine', 'Rock'),
    'MHEALTH': (
        'null',
        'Standing still (1 min)',
        'Sitting and relaxing (1 min)',
        'Lying down (1 min)',
        'Walking (1 min)',
        'Climbing stairs (1 min)',
        'Waist bends forward (20x)',
        'Frontal elevation of arms (20x)',
        'Knees bending (crouching) (20x)',
        'Cycling (1 min)',
        'Jogging (1 min)',
        'Running (1 min)',
        'Jump front & back (20x)'
    )
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


def init(dataset_name, train_prop):
    if not os.path.exists('./data'):
        os.mkdir('./data')

    print("Downloading {}...".format(dataset_name))
    data = request.urlopen(URL[dataset_name])

    if dataset_name is 'Sonar':
        html_response = data.read()
        encoding = data.headers.get_content_charset('utf-8')
        data = html_response.decode(encoding)
        data = data.replace('M', '0')
        data = data.replace('R', '1')

        with open('temp.txt', 'w') as temp:
            temp.write(data)

        data = np.genfromtxt('temp.txt', delimiter=',')
        os.remove('temp.txt')
    elif dataset_name is 'MHEALTH':
        # Rertrive and extract file
        request.urlretrieve(URL[dataset_name], "temp.zip")
        with zipfile.ZipFile('temp.zip', 'r') as zip_ref:
            zip_ref.extractall()

        # Load data for all patients
        patients = [np.genfromtxt('MHEALTHDATASET/mHealth_subject{}.log'
                                  .format(i),
                                  delimiter='	') for i in range(1, 11)]
        data = np.concatenate(patients)

        # Remove temp files
        os.remove('temp.zip')
        shutil.rmtree('MHEALTHDATASET')
    else:
        data = np.genfromtxt(data, delimiter=',')

    np.random.shuffle(data)

    x, y = data[:, :-1], data[:, -1]
    split = int(train_prop * x.shape[0])

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


def load_dataset(dataset_name='MNIST', train_prop=None, one_hot=False):
    """Load dataset as numpy arrays.

    Download dataset if not already available under ./data.
    """
    if dataset_name not in SUPPORTED:
        raise Exception('{} is not supported.'.format(dataset_name))

    if not os.path.exists('data/{}.pkl'.format(dataset_name)):
        if dataset_name in ['MNIST', 'KMNIST', 'Fashion-MNIST']:
            if train_prop is not None:
                print('Warning! MNIST datasets ignore '
                      'the train_prop argument.')
            init_mnist(dataset_name)

        elif dataset_name in ['Banknote', 'Sonar', 'MHEALTH']:
            if train_prop is None:
                train_prop = 0.8
            init(dataset_name, train_prop)

    elif train_prop is not None:
        raise Exception('train_prop should only be used when initializing '
                        'dataset. Please delete the .pkl file in the data '
                        'directory and try again.')

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
