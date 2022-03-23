'''This script was written by Maria Szawerna for the LT2222 V22 Machine Learning course at the University of Gothenburg between 16 and 23.03.2022
    It is intended to be run from the command line together with 3 command line arguments: the input file that the training data is obtained from,
    the output file that the trained model, size od the training data, and the column names will be saved to, and the name of the model that will
    be trained, SVC or NB; nevertheless, the functions and some variables can be imported into other scripts on their own. It requires a file from
    sample.py.'''

import argparse
import pickle
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import pandas as pd
import random
import itertools

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
    # this is done by iterating over all the column names and checking if those exist in the sample and appending a 0 or 1, and appending the consonant to a separate list
    # the samples are also shuffled before this is done in case that would introduce any bias in the machine; returns two lists representing the X and y
    random.shuffle(all_samples) 
    sample_vectors = []
    sample_classes = []
    for sample in all_samples:
        characters = sample[0]
        consonant = sample[1]
        vector = []
        for label in possible_columns:  # determines if there should be a 1 or a 0 in a given column
            if label != 'predicted consonant':
                if label in characters:
                    vector.append(1)
                else:  # if label not in characters
                    vector.append(0)
        sample_vectors.append(vector)
        sample_classes.append(consonant)

    train_X = np.array(sample_vectors)
    train_y = np.array(sample_classes)

    return train_X, train_y, possible_columns

def create_encoded_data(all_samples, possible_columns):
    # performs a role analogous to the function above, but preparing the data for NB, using the OrdinalEncoder class from sklearn to process the data (this was the suggested way of doing that for
    # the documentation and the tutorials for CategoricalNB, however, it also seems to work for MultinomialNB so I decided to leave it as such). The data here is transformed into little lists
    # of four characters+positions each; then from the lists of all such permitted combinations a big list of all their possible products is made, on the basis of which the OrdinalEncoder is trained,
    # and subsequently the training data is encoded by it; returns the encoded data representing the X, the y, and the encoder.
    sample_vectors = []
    sample_classes = []
    for sample in all_samples:
        character1 = sample[0][0]
        character2 = sample[0][1]
        character3 = sample[0][2]
        character4 = sample[0][3]
        consonant = sample[1]
        vector = []

        # makes sure that only the 'permitted' combinations are entered into the vector, i.e. assigns UNK to those that are not known
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

    # initializes the lists that will be construct the array to train OrdinalEncoder on
    characters1 = ['UNK_1']
    characters2 = ['UNK_2']
    characters3 = ['UNK_3']
    characters4 = ['UNK_4']

    for combo in possible_columns:
        if combo[2] == '1':
            characters1.append(combo)
        elif combo[2] == '2':
            characters2.append(combo)
        elif combo[2] == '3':
            characters3.append(combo)
        elif combo[2] == '4':
            characters4.append(combo)
    
    possible_combos = list(itertools.product(characters1, characters2, characters3, characters4))
    possible_combos_lists = []
    for combo in possible_combos:
        new_combo = list(combo)
        possible_combos_lists.append(new_combo)

    enc = OrdinalEncoder()
    enc.fit(possible_combos_lists)  # trains the OrdinalEncoder on all the possible combinations of allowed features
    enc_sample_vectors = enc.transform(sample_vectors)  # transfomrs the samples into that format

    train_X = np.array(enc_sample_vectors)
    train_y = np.array(sample_classes)

    return train_X, train_y, enc


def train(all_samples, possible_columns, model):
    # this function determines which kind of model will be trained and quits the script in case a request is made that cannot be fulfilled; if the selected model is
    # available, it is trained on the given train_X and train_y arrays, and a trained model is returned
    if model == "SVC":
        machine = SVC(kernel='linear', probability=True)
        train_X, train_y, enc = create_vectors(all_samples, possible_columns)
        
    elif model == "NB":
        machine = MultinomialNB()
        train_X, train_y, enc = create_encoded_data(all_samples, possible_columns)
    else:
        print("This is not a permitted model. You can only choose between SVC and NB!")
        quit()
    machine.fit(train_X, train_y)
    
    return machine, enc

def save_model(model, enc, possible_columns, nr_of_samples, filename):
    # this function saves the trained model along with the encoder (in the case of the SVC model it is just another list of all the possible columns), names of the columns 
    # it was trained on and the number of the samples it was trained on into a given file
    with open(filename, 'wb') as model_file:
        pickle.dump((model, enc, possible_columns, nr_of_samples), model_file)

if __name__ == "__main__":
    # here the functions above are used to first obtain the data from the file, then count the samples, create the vectors, create the DataFrame, split it, train the machine, and save it;
    # all of the necessary arguments are obtained from the command line using argparse
    all_samples, possible_columns = import_training_data(args.input_file)
    nr_of_samples = len(all_samples)
    machine, enc = train(all_samples, possible_columns, args.model)
    save_model(machine, enc, possible_columns, nr_of_samples, args.output_file)
    print("Training finished successfully.")
