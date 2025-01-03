# mnist
Example code for training on the MNIST dataset using PyTorch, with CUDA enabled.

The MNIST dataset consists black/white images that display one of ten handwritten numbers (0 to 9). These images are represented by a 28x28 matrix, with each matrix element depicting its corresponding grayscale intensity.

The training set has 60,000 images and the test set has 10,000 separate images.

This repository includes:
* a custom module `mnist_tools.py`, which defines the LeNet5 convolutional neural network, the train function, and the test function
* a jupyter notebook `mnist.ipynb`, which performs training, validation, and testing of the dataset
* a python script `main.py` which *also* performs training, validation, and testing of the dataset, **with the added bonus of user input and an option to save the fully trained model**
