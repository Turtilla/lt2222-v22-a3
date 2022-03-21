'''This script was written by Maria Szawerna for the LT2222 V22 Machine Learning course at the University of Gothenburg between 16 and 21.03.2022
    It is intended to be run from the command line together with 3 command line arguments: the input file that the training data is obtained from,
    the output file that the trained model, size od the training data, and the column names will be saved to, and the name of the model that will
    be trained, SVC or NB; nevertheless, the functions and some variables can be imported into other scripts on their own.'''

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

# this performs the same function here as in sample.py (and perplexity.py), but for the sake of clarity I did not import it but just copied it
consonants_set = set(['b', 'c', 'd', 'f', 'g', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 's', 't', 'v', 'x', 'y', 'z', 'h', 'r', 't', 'w', 'y'])

def import_training_data(filename):
    # this function imports the list of all the training samples and the list of possible columns obtained from sample.py and saved in a pickle file and returns them as a tuple
    with open(filename, 'rb') as training_file:
        all_samples, possible_columns = pickle.load(training_file)

    return all_samples, possible_columns

def create_vectors(all_samples, possible_columns):
    # this function takes a list of samples and a list of possible columns to judge them by and creates a vector for each sample, returning a list of those vectors; 
    # this is done by iterating over all the column names and checking if those exist in the sample and appending a 0 or 1, and finally appending the consonant
    # the samples are also shuffled before this is done in case that would introduce any bias in the machine
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
    # this function takes the list of vectors corresponding to the samples and first turns it into a numpy array which then is converted into a pandas DataFrame, 
    # which is subsequently returned
    vector_array = np.array(sample_vectors)
    vector_df = pd.DataFrame(vector_array)

    return vector_df

def split_samples(fulldf):
    # this function splits the dataframe into the X and y, meaning the vectors and the corresponding classes, those two DataFrames are returned
    column_number = len(fulldf.columns)
    train_X = fulldf.iloc[:, 0:column_number-2]
    train_y = fulldf.iloc[:, column_number-1]

    return train_X, train_y

def train(train_X, train_y, model):
    # this function determines which kind of model will be trained and quits the script in case a request is made that cannot be fulfilled; if the selected model is
    # available, it is trained on the given train_X and train_y DataFrames, and a trained model is returned
    if model == "SVC":
        machine = SVC(kernel='linear', probability=True)
    elif model == "NB":
        machine = CategoricalNB()
    else:
        print("This is not a permitted model. You can only choose between SVC and NB!")
        quit()
    machine.fit(train_X, train_y)
    
    return machine

def save_model(model, possible_columns, nr_of_samples, filename):
    # this function saves the trained model along with the names of the columns it was trained on and the number of the samples it was trained on into a given file
    with open(filename, 'wb') as model_file:
        pickle.dump((model, possible_columns, nr_of_samples), model_file)

if __name__ == "__main__":
    # here the functions above are used to first obtain the data from the file, then count the samples, create the vectors, create the DataFrame, split it, train the machine, and save it;
    # all of the necessary arguments are obtained from the command line using argparse
    all_samples, possible_columns = import_training_data(args.input_file)
    nr_of_samples = len(all_samples)
    sample_vectors = create_vectors(all_samples, possible_columns)
    data_frame = create_df(sample_vectors)
    train_X, train_y = split_samples(data_frame)
    machine = train(train_X, train_y, args.model)
    save_model(machine, possible_columns, nr_of_samples, args.output_file)
    print("Training finished successfully.")
