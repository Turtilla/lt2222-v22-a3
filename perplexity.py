'''This script was written by Maria Szawerna for the LT2222 V22 Machine Learning course at the University of Gothenburg between 16 and 21.03.2022
    It is intended to be run from the command line together with 2 command line arguments: the input file that the model and other data are obtained from,
    the second input file containing the testing data; nevertheless, the functions and some variables can be imported into other scripts on their own.'''

import argparse
import pickle
from sklearn.svm import SVC
from sklearn.naive_bayes import CategoricalNB
import numpy as np
import pandas as pd
import random
import gzip
import math

parser = argparse.ArgumentParser()
parser.add_argument("model_file", help="the file containing the model you want to use")
parser.add_argument("test_file", help="the file you want to take the testing data from; it has to be in a .gz or .txt format")

args = parser.parse_args()

# this performs the same function here as in sample.py, but for the sake of clarity I did not import it but just copied it
consonants_set = set(['b', 'c', 'd', 'f', 'g', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 's', 't', 'v', 'x', 'y', 'z', 'h', 'r', 't', 'w', 'y'])

def import_model(filename):
    # this function is identical to the one in test.py but copied and not imported for the sake of simplicity
    with open(filename, 'rb') as model_file:
        model, possible_columns, nr_of_samples = pickle.load(model_file)

    return model, possible_columns, nr_of_samples

def sample_lines(filename):
    # this function is identical to the one in sample.py but copied and not imported for the sake of simplicity
    if filename.endswith(".gz"):
        unzipped_file = gzip.open(filename, "rb")
        lines = unzipped_file.readlines()
        sampled_lines = []
        for line in lines:
            decoded_line = line.decode('utf8').strip().lower()
            sampled_lines.append(decoded_line)
        unzipped_file.close()
        return sampled_lines

    elif filename.endswith(".txt"):
        opened_file = open(filename, "r")
        lines = opened_file.readlines()
        sampled_lines = []
        for line in lines:
            stripped_line = line.strip().lower()
            sampled_lines.append(stripped_line)
        opened_file.close()
        return sampled_lines

    else:
        print("This type of file is not allowed!")
        quit()

def find_closest_consonant(line):
    # this function is identical to the one in sample.py but copied and not imported for the sake of simplicity
    for character in line:
        if character in consonants_set:
            return character
        else:
            continue
    return False

def create_samples(sampled_lines):
    # this function is identical to the one in sample.py but copied and not imported for the sake of simplicity
    all_samples = []
    for line in sampled_lines:
        for i in range(0, len(line)-5):
            four_characters = (line[i]+"_1", line[i+1]+"_2", line[i+2]+"_3", line[i+3]+"_4")
            closest_consonant = find_closest_consonant(line[i+4:])
            if closest_consonant != False:
                full_sample = (four_characters, closest_consonant)
                all_samples.append(full_sample)
    return all_samples  

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

def return_probabilities(model, test_X):
    # this function takes the model and a DataFrame of vectors without the classes and returns the probabilities predicted by the model; since
    # sklearn uses by default the np.log() function to return log probabilities, these need to be converted from base e to base 2 before being
    # used in the perplexity formula; this is also done within this function and an array of log2 probabilities is returned
    probabilities = model.predict_log_proba(test_X)
    base_2_logs = np.divide(probabilities, np.log(2))

    return base_2_logs

def retrieve_correct_probs(model, test_X, test_y, nr_of_samples):
    # this function takes the trained model, test vectors, corresponding test classes, and the number of samples and first retrieves the log2 probabilities
    # using the return_probabilities() function, then retrieves the corresponding classes contained within the model, and then based on that and the list of
    # true classification retrieves the probability assigned by the model to the "correct" answer; in case none such probability exists, a simplified 
    # version of add-1/Laplace smoothing is implemented (as suggested by Asad Sayeed) where the log2 of 1 divided by the number of training samples is assigned
    # to it instead; a list of probabilities associated with the correct classification is returned
    correct_probs = []
    probabilities = return_probabilities(model, test_X)
    classes = model.classes_
    for i in range(0,len(test_y)):
        correct_ans = test_y[i]
        probs = probabilities[i]
        for j in range(0,len(classes)):
            if classes[j] == correct_ans:
                correct_prob = probs[j]
                correct_probs.append(correct_prob)
        if correct_ans not in classes:  # if there is no probability for the correct answer according to the model
            correct_prob = 1/nr_of_samples  # accounting for 0 probabilities
            correct_prob_log = math.log(correct_prob,2)  # base 2 logarithm
            correct_probs.append(correct_prob_log)

    return correct_probs

def perplexity(model, test_X, test_y, nr_of_samples):
    # this function takes the trained model, the test vectors, corresponding test classes, and the number of samples, uses the retrieve_correct_probs() function to
    # obtain a list of log2 probabilities associated with the correct answers, compares the length of that to the length of the true answers, prints 'inf' if they are
    # uneven (which should not happen as smoothing is built in), and otherwise calculates the perplexity by raising the base 2 to the power of negative one over the
    # number of test samples times the sum of the log2 probabilities predicted by the model;
    correct_probs = retrieve_correct_probs(model, test_X, test_y, nr_of_samples)
    if len(correct_probs) != len(test_y):
        print("inf")
    else:
        sample_number = len(test_X)
        base = 2
        exponent = -(1/sample_number)*(sum(correct_probs))
        pp = math.pow(base, exponent)
        print(f'The perplexity of this {model} model is equal to {pp}')



if __name__ == "__main__":
    model, possible_columns, nr_of_samples = import_model(args.model_file)

    sampled_lines = sample_lines(args.test_file)
    full_samples = create_samples(sampled_lines)
    sample_vectors = create_vectors(full_samples, possible_columns)
    data_frame = create_df(sample_vectors)
    test_X, test_y = split_samples(data_frame)

    perplexity(model, test_X, test_y, nr_of_samples)