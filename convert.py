# MIT License

# Copyright (c) 2017 Hyeonseok Jung

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.



# ALL RIGHTS TO ORIGINAL OWNER 
# https://github.com/hsjeong5/MNIST-for-Numpy

# Modified by Steven Furgurson
# https://github.com/theDweeb
import mnist
import numpy as np
import imageio
import os.path
from PIL import Image

# Check if MNIST dataset has been downloaded
isDownloaded = (
    os.path.exists("train-images-idx3-ubyte.gz") and
    os.path.exists("train-images-idx3-ubyte.gz") and
    os.path.exists("train-images-idx3-ubyte.gz") and
    os.path.exists("train-images-idx3-ubyte.gz")
)

if(not isDownloaded):
    print("Downloading files")
    mnist.init()

print("Files downloaded")

# Hold MNIST dataset
# x_train: Training images
# t_train: Training labels
# x_test: Testing images
# t_test = Testing labels
x_train, t_train, x_test, t_test = mnist.load()

# Pull a image to test for MLP
# Just adjust the left index
image = x_test[0,:]
image.resize((28,28))

# For my MLP I used np.int32 for image.
# Make sure the data type matches for the vector<dType>
npimage = np.asarray(image, dtype=np.int32)

np.save("image", npimage)

# Shows image for labeling
img = Image.fromarray(image)
img.save('image.png')
img.show()