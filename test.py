'''This script was written by Maria Szawerna for the LT2222 V22 Machine Learning course at the University of Gothenburg between 16 and 21.03.2022
    It is intended to be run from the command line together with 3 command line arguments: the input file that the model and other data are obtained from,
    the second input file containing the testing data, and the type of average that will be used for some of the measures; 
    nevertheless, the functions and some variables can be imported into other scripts on their own.'''

import argparse
import pickle
from sklearn.svm import SVC
from sklearn.naive_bayes import CategoricalNB
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
        model, possible_columns, nr_of_samples = pickle.load(model_file)

    return model, possible_columns, nr_of_samples

def create_vectors(all_samples, possible_columns):
    # this function is identical to the one in train.py but copied and not imported for the sake of simplicity
    random.shuffle(all_samples) 
    sample_vectors = []
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
        vector.append(consonant)
        sample_vectors.append(vector)

    return sample_vectors

def create_df(sample_vectors):
    # this function is identical to the one in train.py but copied and not imported for the sake of simplicity
    vector_array = np.array(sample_vectors)
    vector_df = pd.DataFrame(vector_array)

    return vector_df

def split_samples(fulldf):
    # this function is identical to the one in train.py but copied and not imported for the sake of simplicity
    column_number = len(fulldf.columns)
    test_X = fulldf.iloc[:, 0:column_number-2]
    test_y = fulldf.iloc[:, column_number-1]

    return test_X, test_y

def eval_model(model, test_X, test_y, average_type):
    # rhis function evaluates what kind of average is requested, and if it is one of the permitted types it evaluates the model on the testing data and prints to the terminal
    # the sklearn metrics accuracy, precision, recall, and f1 score, the latter three averaged using the requested average type
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
    model, possible_columns, nr_of_samples = import_model(args.model_file)
    test_samples = import_testing_data(args.test_file)
    test_vectors = create_vectors(test_samples, possible_columns)
    data_frame = create_df(test_vectors)
    test_X, test_y = split_samples(data_frame)
    eval_model(model, test_X, test_y, args.average_type)
