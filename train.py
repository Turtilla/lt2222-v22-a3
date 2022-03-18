import argparse
import pickle
from sklearn.svm import SVC
from sklearn.naive_bayes import CategoricalNB
import numpy as np
import pandas as pd
import random

parser = argparse.ArgumentParser()
parser.add_argument("input_file", help="the file you want to import training data from")
parser.add_argument("output_file", help="the file you want to save the trained model to")
parser.add_argument("model", help="the model you want to use: choose between SVC and NB")

args = parser.parse_args()

consonants_set = set(['b', 'c', 'd', 'f', 'g', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 's', 't', 'v', 'x', 'y', 'z', 'h', 'r', 't', 'w', 'y'])

def import_training_data(filename):
    with open(filename, 'rb') as training_file:
        all_samples, possible_columns = pickle.load(training_file)

    return all_samples, possible_columns

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

    ### potential padding?
    for consonant in consonants_set:
        vector = []
        for label in possible_columns:
            if label != 'predicted consonant':
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
    train_X = fulldf.iloc[:, 0:column_number-2]
    train_y = fulldf.iloc[:, column_number-1]

    return train_X, train_y

def train(train_X, train_y, model):
    if model == "SVC":
        machine = SVC(kernel='linear', probability=True)
    elif model == "NB":
        machine = CategoricalNB()
    else:
        print("This is not a permitted model. You can only choose between SVC and NB!")
        quit()
    machine.fit(train_X, train_y)
    
    return machine

def save_model(model, possible_columns, filename):
    with open(filename, 'wb') as model_file:
        pickle.dump((model, possible_columns), model_file)

if __name__ == "__main__":
    all_samples, possible_columns = import_training_data(args.input_file)
    sample_vectors = create_vectors(all_samples, possible_columns)
    data_frame = create_df(sample_vectors)
    train_X, train_y = split_samples(data_frame)
    machine = train(train_X, train_y, args.model)
    save_model(machine, possible_columns, args.output_file)
