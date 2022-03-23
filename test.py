'''This script was written by Maria Szawerna for the LT2222 V22 Machine Learning course at the University of Gothenburg between 16 and 23.03.2022
    It is intended to be run from the command line together with 3 command line arguments: the input file that the model and other data are obtained from,
    the second input file containing the testing data, and the type of average that will be used for some of the measures; 
    nevertheless, the functions and some variables can be imported into other scripts on their own. It requires files from sample.py and train.py.'''

import argparse
import pickle
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import pandas as pd
import sklearn.metrics as skm
import random

parser = argparse.ArgumentParser()
parser.add_argument("model_file", help="the file you want to import the model from")
parser.add_argument("test_file", help="the file you want to import the testing data from")
parser.add_argument("average_type", help="the kind of average that you want implemented in the scores; choose between micro or macro")

args = parser.parse_args()

def import_testing_data(filename):
    # this function imports the list of all the testing samples obtained from sample.py and saved in a pickle file and returns them as a tuple
    with open(filename, 'rb') as testing_file:
        test_samples = pickle.load(testing_file)

    return test_samples

def import_model(filename):
    # this function imports the trained model, the list of possible columns obtained from train.py, and the number of training samples saved in a pickle file and returns them as a tuple
    with open(filename, 'rb') as model_file:
        model, enc, possible_columns, nr_of_samples = pickle.load(model_file)

    return model, enc, possible_columns, nr_of_samples

def create_vectors(all_samples, possible_columns):
    # works the same as the function in train.py
    random.shuffle(all_samples) 
    sample_vectors = []
    sample_classes = []
    for sample in all_samples:
        characters = sample[0]
        consonant = sample[1]
        vector = []
        for label in possible_columns:
            if label != 'predicted consonant':
                if label in characters:
                    vector.append(1)
                else:  # if label not in characters
                    vector.append(0)
        sample_vectors.append(vector)
        sample_classes.append(consonant)

    test_X = np.array(sample_vectors)
    test_y = np.array(sample_classes)

    return test_X, test_y

def create_encoded_data(all_samples, enc):
    # works similarly to the function in train.py, but does not traint the encoder all over again, instead uses the one provided together with the model from train.py
    sample_vectors = []
    sample_classes = []
    for sample in all_samples:
        character1 = sample[0][0]
        character2 = sample[0][1]
        character3 = sample[0][2]
        character4 = sample[0][3]
        consonant = sample[1]
        vector = []

        if character1 in possible_columns:
            vector.append(character1)
        else:
            vector.append('UNK_1')

        if character2 in possible_columns:
            vector.append(character2)
        else:
            vector.append('UNK_2')

        if character3 in possible_columns:
            vector.append(character3)
        else:
            vector.append('UNK_3')

        if character4 in possible_columns:
            vector.append(character4)
        else:
            vector.append('UNK_4')

        sample_vectors.append(vector)
        sample_classes.append(consonant)

    enc_sample_vectors = enc.transform(sample_vectors)  # uses the imported encoder, already pre-trained

    train_X = np.array(enc_sample_vectors)
    train_y = np.array(sample_classes)

    return train_X, train_y

def eval_model(model, all_samples, possible_columns, enc, average_type):
    # this function evaluates what kind of average is requested, and if it is one of the permitted types it evaluates the model on the testing data and prints to the terminal
    # the sklearn metrics accuracy, precision, recall, and f1 score, the latter three averaged using the requested average type
    if isinstance(model, MultinomialNB):
        test_X, test_y = create_encoded_data(all_samples, enc)
    elif isinstance(model, SVC):
        test_X, test_y = create_vectors(all_samples, possible_columns)
    else:
        print('This kind of model is not supported! Choose a file with an SVC or MultinomialNB model.')
        quit()

    if average_type == 'micro' or average_type == 'macro':
        pred_y = model.predict(test_X)
        accuracy = skm.accuracy_score(test_y, pred_y)
        precision = skm.precision_score(test_y, pred_y, average=average_type, zero_division=0)
        recall = skm.recall_score(test_y, pred_y, average=average_type,  zero_division=0)
        f1 = skm.f1_score(test_y, pred_y, average=average_type,  zero_division=0)
    else:
        print('This type of averaging is not accepted! Choose between micro or macro.')
        quit()

    print(f"For the model {model} the following scores were obtained with {average_type}-averaging (wherever applicable): \n\taccuracy = {accuracy} \n\tprecision = {precision} \n\trecall = {recall} \n\tf1 score = {f1}")

if __name__ == "__main__":
    # here the functions above are used to first retrieve the trained model and the data packed together with it, then retrieve the testing data and process it in a similar fashion, and finally to evaluate the model;
    # all of the necessary arguments are obtained from the command line using argparse
    model, enc, possible_columns, nr_of_samples = import_model(args.model_file)
    test_samples = import_testing_data(args.test_file)
    print(model)
    eval_model(model, test_samples, possible_columns, enc, args.average_type)
