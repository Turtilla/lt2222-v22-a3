'''This script was written by Maria Szawerna for the LT2222 V22 Machine Learning course at the University of Gothenburg between 16 and 23.03.2022
    It is intended to be run from the command line together with 2 command line arguments: the input file that the model and other data are obtained from,
    the second input file containing the testing data; nevertheless, the functions and some variables can be imported into other scripts on their own.
    It requires a file from train.py and a file to test the perplexity of.'''

import argparse
import pickle
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import OrdinalEncoder
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
        model, enc, possible_columns, nr_of_samples = pickle.load(model_file)

    return model, enc, possible_columns, nr_of_samples

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

    enc_sample_vectors = enc.transform(sample_vectors)

    train_X = np.array(enc_sample_vectors)
    train_y = np.array(sample_classes)

    return train_X, train_y

def return_probabilities(model, test_X):
    # this function takes the model and a DataFrame of vectors without the classes and returns the probabilities predicted by the model; since
    # sklearn uses by default the np.log() function to return log probabilities, these need to be converted from base e to base 2 before being
    # used in the perplexity formula; this is also done within this function and an array of log2 probabilities is returned
    probabilities = model.predict_log_proba(test_X)
    base_2_logs = np.divide(probabilities, np.log(2))

    return base_2_logs

def retrieve_correct_probs_svc(model, test_X, test_y, nr_of_samples):
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

def retrieve_correct_probs_nb(model, test_X, test_y, nr_of_samples):
    # this function works analogously to the one above, but is separate due to some peculiarities of the NB models and the errors that they threw; here the probabilities
    # are checked vector by vector and not for the whole array at once. The rest follows the same pattern of checking what probability is associated with the correct answer
    # and calculating the log2 probability for it
    probabilities = []
    for vector in test_X:
        probs = model.predict_log_proba(np.array(vector).reshape(1,-1))
        probabilities.append(probs)
    
    correct_probs = []
    classes = model.classes_
    for i in range(0,len(test_y)):
        correct_ans = test_y[i]
        probs = probabilities[i]

        for j in range(0,len(classes)):
            if classes[j] == correct_ans:
                correct_prob = probs[0][j]
                correct_probs.append(correct_prob)
        if correct_ans not in classes:  # if there is no probability for the correct answer according to the model
            correct_prob = 1/nr_of_samples  # accounting for 0 probabilities
            correct_prob_log = math.log(correct_prob,2)  # base 2 logarithm
            correct_probs.append(correct_prob_log)

    return correct_probs

def perplexity(model, all_samples, possible_columns, enc, nr_of_samples):
    # this function takes the trained model, the test vectors, corresponding test classes, and the number of samples, uses the retrieve_correct_probs() function to
    # obtain a list of log2 probabilities associated with the correct answers, compares the length of that to the length of the true answers, prints 'inf' if they are
    # uneven (which should not happen as smoothing is built in), and otherwise calculates the perplexity by raising the base 2 to the power of negative one over the
    # number of test samples times the sum of the log2 probabilities predicted by the model;
    if isinstance(model, MultinomialNB):
        test_X, test_y = create_encoded_data(all_samples, enc)
        correct_probs = retrieve_correct_probs_nb(model, test_X, test_y, nr_of_samples)
    elif isinstance(model, SVC):
        test_X, test_y = create_vectors(all_samples, possible_columns)
        correct_probs = retrieve_correct_probs_svc(model, test_X, test_y, nr_of_samples)
    else:
        print('This kind of model is not supported! Choose a file with an SVC or MultinomialNB model.')
        quit()

    if len(correct_probs) != len(test_y):
        print("inf")
    else:
        sample_number = len(test_X)
        base = 2
        exponent = -(1/sample_number)*(sum(correct_probs))
        pp = math.pow(base, exponent)
        print(f'The perplexity of this {model} model is equal to {pp}')

if __name__ == "__main__":
    model, enc, possible_columns, nr_of_samples = import_model(args.model_file)

    sampled_lines = sample_lines(args.test_file)
    full_samples = create_samples(sampled_lines)

    perplexity(model, full_samples[1600:1800], possible_columns, enc, nr_of_samples)