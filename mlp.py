'''
Desc: Multilayer Perceptron implementation in Python
Date: 26.09.2018
Author: Manohar Mukku
GitHub: https://github.com/manoharmukku/multilayer-perceptron-in-python
'''

import sys
import getopt
import pandas as pd

def main(argv):
    '''
    Main function which controls the flow of the program
    Arguments: argv - command line arguments and flags
    '''

    # Get the command line arguments
    try:
        opts, args = getopt.getopt(argv, "hl:m:b:r:f:s:i:", ["help", "lr=", "momentum=", "batchsize=", "regularization=", "file=", "seed=", "iterations="])
    except getopt.GetoptError:
        sys.exit(2)

    # Defaults
    lr = 0.01
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
        elif (opt in ["-m", "--momentum"]):
            momentum = arg
        elif (opt in ["-b", "--batchsize"]):
            batch_size = arg
        elif (opt in ["-r", "--regularization"]):
            regularization = arg
        elif (opt in ["-f", "--file"]):
            file = arg
        elif (opt in ["-s", "--seed"]):
            seed = arg
        elif (opt in ["-i", "--iterations"]):
            max_iter = arg
        else:
            sys.exit("Invalid arguments. See help by running mlp.py --help")

    # Sanity check learning rate
    try:
        lr = float(lr)
    except ValueError:
        sys.exit("Oops! Plese enter a float value for learning rate")
    if (lr <= 0):
        sys.exit("Oops! Please enter a positive float value for momentum")

    # Sanity check momentum
    try:
        momentum = float(momentum)
    except ValueError:
        sys.exit("Oops! Please enter a float value for momentum")
    if (momentum < 0):
        sys.exit("Oops! Please enter a non-negative float value for momentum")

    # Sanity check batch size
    try:
        batch_size = int(batch_size)
    except ValueError:
        sys.exit("Oops! Please enter an integer value for batch size")
    if (batch_size <= 0):
        sys.exit("Oops! Please enter a positive integer value for batch size")

    # Sanity check regularization
    try:
        regularization = float(regularization)
    except ValueError:
        sys.exit("Oops! Please enter a float value for regularization parameter")
    if (regularization < 0):
        sys.exit("Oops! Please enter a non-negative float value for regularization parameter")

    # Sanity check file
    with open(file) as data_file:
        df = pd.read_csv(file)

    # Sanity check seed
    try:
        seed = int(seed)
    except ValueError:
        sys.exit("Oops! Enter an integer value for seed")

    # Sanity check for iterations
    try:
        max_iter = int(max_iter)
    except ValueError:
        sys.exit("Oops! Please enter an integer value for iterations")
    if (max_iter <= 0):
        sys.exit("Oops! Please enter a positive integer value for iterations")


if __name__ == "__main__":
    main(sys.argv[1:])