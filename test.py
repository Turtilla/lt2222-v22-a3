import argparse
import pickle
from sklearn.svm import SVC
from sklearn.naive_bayes import CategoricalNB
import random
import numpy as np
import pandas as pd
import math

parser = argparse.ArgumentParser()
parser.add_argument("model_file", help="the file you want to import the model from")
parser.add_argument("test_file", help="the file you want to import the testing data from")

args = parser.parse_args()

def import_testing_data(filename):
    with open(filename, 'rb') as testing_file:
        test_samples = pickle.load(testing_file)

    return test_samples

def import_model(filename):
    with open(filename, 'rb') as model_file:
        model = pickle.load(model_file)

    return model

if __name__ == "__main__":
    model = import_model(args.model_file)
    test_samples = import_testing_data(args.test_file)