# mnist
Example code for training on the MNIST dataset using PyTorch, with CUDA enabled.

The MNIST dataset consists of black/white images that display one of ten handwritten numbers (0 to 9). These images are represented by a 28x28 matrix, with each matrix element depicting its corresponding grayscale intensity.

The training set has 60,000 images and the test set has 10,000 images.

This repository includes:
* a custom module `mnist_tools.py`, which defines the LeNet5 convolutional neural network, my custom convolutional network, the train function, and the test function
* a jupyter notebook `mnist.ipynb`, which performs training, validation, and testing of the dataset (using LeNet5)
* a python script `lenet_main.py` which *also* performs training, validation, and testing of the dataset, **with the added bonus of user input and an option to save the fully trained model**
* a python script `custom_main.py` &mdash; similar to `lenet_main.py`, except using a custom CNN that I developed
  * achieves similar accuracy of ~98% for a batch size of 32, trained through 5 epochs

The LeNet5 architecture was based off this wikipedia article: https://en.wikipedia.org/wiki/LeNet
