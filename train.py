import argparse
import pickle
from sklearn.svm import SVC
from sklearn.naive_bayes import CategoricalNB
import random
import numpy as np
import pandas as pd
import math

parser = argparse.ArgumentParser()
parser.add_argument("input_file", help="the file you want to import training data from")
parser.add_argument("output_file", help="the file you want to save the trained model to")
parser.add_argument("model", help="the model you want to use: choose between SVC and NB")

args = parser.parse_args()

def import_training_data(filename):
    with open(filename, 'rb') as training_file:
        all_samples = pickle.load(training_file)

    return all_samples

def create_df(all_samples):
    possible_columns = []
    possible_columns_set = set()
    for sample in all_samples:
        characters = sample[0]
        for character in characters:
            if character not in possible_columns_set:
                possible_columns.append(character)
                possible_columns_set.add(character)
    possible_columns.append('predicted consonant')
    
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

    vector_array = np.array(sample_vectors)
    vector_df = pd.DataFrame(vector_array, columns=possible_columns)

    return vector_df

def split_samples(fulldf):
    column_number = len(fulldf.columns)
    train_X = fulldf.iloc[:, 0:column_number-2]
    train_y = fulldf.iloc[:, column_number-1]

    return train_X, train_y

def train(train_X, train_y, model):
    if model == "SVC":
        machine = SVC(kernel='linear')
    elif model == "NB":
        machine = CategoricalNB()
    else:
        print("This is not a permitted model. You can only choose between SVC and NB!")
        quit()
    machine.fit(train_X, train_y)
    
    return machine

def save_model(model, filename):
    with open(filename, 'wb') as model_file:
        pickle.dump(model, model_file)

if __name__ == "__main__":
    all_samples = import_training_data(args.input_file)
    data_frame = create_df(all_samples)
    train_X, train_y = split_samples(data_frame)
    machine = train(train_X, train_y, args.model)
    save_model(machine, args.output_file)
