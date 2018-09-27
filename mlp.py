'''
Author: Manohar Mukku
Date: 26.09.2018
Description: Multilayer Perceptron implementation in Python
GitHub: https://github.com/manoharmukku/multilayer-perceptron-in-python
'''

import sys
import getopt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import oneHotEncoder

class MLP(object):
    def __init__(self, lr, sizes, activations, error_function, momentum, batch_size, regularization, seed, max_iter):
        self.lr = lr
        self.sizes = sizes
        self.activations = activations
        self.error_function = error_function
        self.momentum = momentum
        self.batch_size = batch_size
        self.regularization = regularization
        self.seed = seed
        self.max_iter = max_iter
        self.weights = None
        self.threshold = 0.01

    def fit(self, X, y):
        '''
        Fits self.weights using MLP training
        # Arguments:
            X: Input
            y: Desired Output
        '''

        # Initialize weights
        self.initialize_weights()




def get_arguments(argv):
    '''
    Get command line arguments using getopt
    Argument: argv which contains arguments and flags
    '''

    # Get the command line arguments
    try:
        opts, args = getopt.getopt(argv, "hl:s:a:e:m:b:r:f:x:i:", ["help", "lr=", "sizes=", "activations=", "error_function=", \
            "momentum=", "batchsize=", "regularization=", "file=", "seed=", "iterations="])
    except getopt.GetoptError:
        sys.exit(2)

    # Defaults
    lr = 0.01
    sizes = "10,3,3,2"
    activations = "identity,sigmoid,sigmoid,sigmoid"
    error_function = "mse"
    momentum = 0
    batch_size = 1
    regularization = 0
    file = "data.csv"
    df = None
    seed = 7
    max_iter = 1500

    # Parse the command line arguments
    for opt, arg in opts:
        if (opt in ["-h", "--help"]):
            usage()
            sys.exit()
        elif (opt in ["-l", "--lr"]):
            lr = arg
        elif (opt in ["-s", "--sizes"]):
            sizes = arg
        elif (opt in ["-a", "--activations"]):
            activations = arg
        elif (opt in ["-e", "--error_function"]):
            error_function = arg
        elif (opt in ["-m", "--momentum"]):
            momentum = arg
        elif (opt in ["-b", "--batchsize"]):
            batch_size = arg
        elif (opt in ["-r", "--regularization"]):
            regularization = arg
        elif (opt in ["-f", "--file"]):
            file = arg
        elif (opt in ["-x", "--seed"]):
            seed = arg
        elif (opt in ["-i", "--iterations"]):
            max_iter = arg
        else:
            sys.exit("Invalid arguments. See help by running mlp.py --help")

    # Sanity check learning rate
    try:
        lr = float(lr)
    except ValueError:
        sys.exit("Oops! Learning rate should be a float value")
    if (lr <= 0):
        sys.exit("Oops! Leraning rate should be positive")

    # Sanity check sizes
    try:
        sizes = [int(n) for n in sizes.split(",")]
    except ValueError:
        sys.exit("Oops! Layer sizes should be integer values")
    for size in sizes:
        if (size <= 0):
            sys.exit("Oops! Layer sizes should be positive integer values")

    # Sanity check activations
    activations = [act for act in activations.split(",")]
    activations = ["identity"].extend(activations)
    for act in activations:
        if (act not in ["identity", "sigmoid", "tanh", "relu", "leakyrelu", "softmax"]):
            sys.exit("Oops! Supported activation functions are only identity, sigmoid, tanh, relu, leakyrelu, softmax")

    # Sanity check error function
    if (error_function not in ["mse"]):
        sys.exit("Oops! Invalid error function")

    # Sanity check momentum
    try:
        momentum = float(momentum)
    except ValueError:
        sys.exit("Oops! Momentum should be a float value")
    if (momentum < 0):
        sys.exit("Oops! Momentum should be non-negative")

    # Sanity check batch size
    try:
        batch_size = int(batch_size)
    except ValueError:
        sys.exit("Oops! Batch size should be an integer value")
    if (batch_size <= 0):
        sys.exit("Oops! Batch size should be positive")

    # Sanity check regularization
    try:
        regularization = float(regularization)
    except ValueError:
        sys.exit("Oops! Regularization should be a float value")
    if (regularization < 0):
        sys.exit("Oops! Regularization should be non-negative")

    # Sanity check file
    with open(file) as data_file:
        df = pd.read_csv(data_file)

    # Sanity check seed
    try:
        seed = int(seed)
    except ValueError:
        sys.exit("Oops! Random seed should be an integer value")

    # Sanity check for iterations
    try:
        max_iter = int(max_iter)
    except ValueError:
        sys.exit("Oops! Number of iterations should be an integer value")
    if (max_iter <= 0):
        sys.exit("Oops! Number of iterations should be positive")

    return lr, sizes, activations, error_function, momentum, batch_size, regularization, df, seed, max_iter

def main(argv):
    '''
    Main function which controls the flow of the program
    Arguments:
        argv - command line arguments with flags
    '''

    # Get the parameters from the command line
    lr, sizes, activations, error_function, momentum, batch_size, regularization, df, seed, max_iter = get_arguments(argv)

    # Create an instance of MLP class
    model = MLP(lr=lr, sizes=sizes, activations=activations, error_function=error_function, momentum=momentum, batch_size=batch_size, \
        regularization=regularization, seed=seed, max_iter=max_iter)

    # Convert the pandas dataframe to numpy array
    data = df.values
    data = data.astype(float)

    # Obtain X from data
    X = data[:, 0:-1]

    # Obtain y from data and onehot
    y = data[:, -1]
    onehot = oneHotEncoder(sparse=False)
    y = onehot.fit_transform(y)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

    # Train the model
    model.fit(X_train, y_train)

    # Predict on X_test
    y_pred = model.predict(X_test)

    # Print the confusion matrix
    model.confusion(y_pred, y_test)

if __name__ == "__main__":
    main(sys.argv[1:])