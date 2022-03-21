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

consonants_set = set(['b', 'c', 'd', 'f', 'g', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 's', 't', 'v', 'x', 'y', 'z', 'h', 'r', 't', 'w', 'y'])

def import_model(filename):
    with open(filename, 'rb') as model_file:
        model, possible_columns = pickle.load(model_file)

    return model, possible_columns

def sample_lines(filename):
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
    for character in line:
        if character in consonants_set:
            return character
        else:
            continue
    return False

def create_samples(sampled_lines):
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

def return_probabilities(model, sample_vectors):
    probabilities = model.predict_log_proba(sample_vectors)
    return probabilities

### experimenting
def retrieve_correct_probs(model, test_X, test_y):
    correct_probs = []
    probabilities = model.predict_log_proba(test_X)
    classes = model.classes_
    for i in range(0,len(test_y)):
        correct_ans = test_y[i]
        probs = probabilities[i]
        for j in range(0,len(classes)):
            if classes[j] == correct_ans:
                correct_prob = probs[j]
                correct_probs.append(correct_prob)
            if correct_ans not in classes:  # if there is no probability for the correct answer according to the model
                correct_prob = 1/len(test_y)  # accounting for 0 probabilities
                correct_prob_log = math.log(correct_prob)
                correct_probs.append(correct_prob_log)

    return correct_probs

def perplexity(model, test_X, test_y):
    correct_probs = retrieve_correct_probs(model, test_X, test_y)
    if len(correct_probs) != len(test_y):
        return "inf"
    else:
        sample_number = len(test_X)
        exponent = -(1/sample_number)*(sum(correct_probs))
        pp = np.exp(exponent)  # because scikit has all the log probabilities in base e, the formula for that is from here https://stats.stackexchange.com/questions/10302/what-is-perplexity
        return pp



if __name__ == "__main__":
    model, possible_columns = import_model(args.model_file)

    sampled_lines = sample_lines(args.test_file)
    full_samples = create_samples(sampled_lines)
    sample_vectors = create_vectors(full_samples, possible_columns)
    data_frame = create_df(sample_vectors)
    test_X, test_y = split_samples(data_frame)
    probabilities = return_probabilities(model, test_X)
    print(model.classes_)
    print(probabilities)
    print(len(data_frame))
    print(len(probabilities))
    print(len(model.classes_))
    print(len(probabilities[0]))
    
    correct_probs = retrieve_correct_probs(model, test_X, test_y)
    print(len(correct_probs))

    pp = perplexity(model, test_X, test_y)
    print(pp)