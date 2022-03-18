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
    with open(filename, 'rb') as testing_file:
        test_samples = pickle.load(testing_file)

    return test_samples

def import_model(filename):
    with open(filename, 'rb') as model_file:
        model, possible_columns = pickle.load(model_file)

    return model, possible_columns

def create_vectors(all_samples, possible_columns):
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

    vector_array = np.array(sample_vectors)
    vector_df = pd.DataFrame(vector_array)

    return vector_df

def split_samples(fulldf):
    column_number = len(fulldf.columns)
    test_X = fulldf.iloc[:, 0:column_number-2]
    test_y = fulldf.iloc[:, column_number-1]

    return test_X, test_y

def eval_model(model, test_X, test_y, average_type):
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
    model, possible_columns = import_model(args.model_file)
    test_samples = import_testing_data(args.test_file)
    test_vectors = create_vectors(test_samples, possible_columns)
    data_frame = create_df(test_vectors)
    test_X, test_y = split_samples(data_frame)
    eval_model(model, test_X, test_y, args.average_type)

    print(len(possible_columns))
