'''This script was written by Maria Szawerna for the LT2222 V22 Machine Learning course at the University of Gothenburg between 16 and 21.03.2022
    It is intended to be run from the command line together with 5 command line arguments: the name of or path to the input file, the desired number
    of samples, the cut-off point for the test/train split of the data, the name of the file to save the test data to, and the name of the file to
    save the training data and the names of the columns to; nevertheless, the functions and some variables can be imported into other scripts on their own.'''

import argparse
import gzip
import random
import math
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("input_file", help="the file you want to make the samples out of (.txt or .gzip)")
parser.add_argument("samples", help="the number of samples that you want to produce", type=int)
parser.add_argument("split", help="the percentage of the samples that will form the test set (e.g. entering 20 here will produce a 20/80 test/train split)", type=int)
parser.add_argument("test_file", help="the file you want to save the test samples to")
parser.add_argument("train_file", help="the file you want to save the training samples to")

args = parser.parse_args()

# the set of all the characters that will be predicted for - or, in other words, the classes
consonants_set = set(['b', 'c', 'd', 'f', 'g', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 's', 't', 'v', 'x', 'y', 'z', 'h', 'r', 't', 'w', 'y'])

def sample_lines(filename):
    # this function detects what kind of file is fed into it and how to open it, quits the script if the extension is not supported; it returns the file as a list where every element is a line from the file
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
    # this function detects the first consonant in the line (based on the consonant set above), and returns it, or a boolean False if there is no consonant in the given string
    for character in line:
        if character in consonants_set:
            return character
        else:
            continue
    return False

def create_samples(sampled_lines, sample_number):
    # this function creates samples out of the lines, based on a given list of strings and number of expected samples; it randomizes the point from which the samples start to be taken
    # the samples appear in the ((character_1, character_2, character_3, character_4), consonant) format; returns a list of such samples containing the given number of samples
    random_slice = random.randint(0, (len(sampled_lines)-sample_number))
    all_samples = []
    for line in sampled_lines[random_slice:]:
        for i in range(0, len(line)-5):
            four_characters = (line[i]+"_1", line[i+1]+"_2", line[i+2]+"_3", line[i+3]+"_4")
            closest_consonant = find_closest_consonant(line[i+4:])
            if closest_consonant != False:
                full_sample = (four_characters, closest_consonant)
                all_samples.append(full_sample)
            if len(all_samples) == sample_number:
                return all_samples  

def create_columns(all_samples):
    # this function creates a list of possible columns or classes for future reference (given how the vectors are created based only on the characters and positions that appear in the initial data,
    # and not all the possible combinations). Returns an ordered list representing the columns of the upcoming dataframe
    possible_columns = []
    possible_columns_set = set()
    for sample in all_samples:
        characters = sample[0]
        for character in characters:
            if character not in possible_columns_set:
                possible_columns.append(character)
                possible_columns_set.add(character)
    possible_columns.append('predicted consonant')
    
    return possible_columns

def split_samples(all_samples, test_percent):
    # this function splits the samples into a test set and a training set based on a list of samples and a given cutoff point which represents how much of the samples will constitute the training set
    cutoff = math.ceil(len(all_samples) * (test_percent / 100))
    test_set = all_samples[:cutoff]
    train_set = all_samples[cutoff:]
    return test_set, train_set

def save_samples(test_set, train_set, possible_columns, test_file, train_file):
    # this function saves the test set and the training set (along with the possible columns) by using pickle into given files
    with open(test_file, 'wb') as testing_file:
        pickle.dump(test_set, testing_file)
    with open(train_file, 'wb') as training_file:
        pickle.dump((train_set, possible_columns), training_file)


if __name__ == "__main__":
    # here the functions above are used to first read from the file, then create samples out of that, define the possible columns, split the samples into the test set and training set, and then save the samples;
    # all of the necessary arguments are obtained from the command line using argparse
    lines = sample_lines(args.input_file)
    full_samples = create_samples(lines, args.samples)
    possible_columns = create_columns(full_samples)
    test_set, train_set = split_samples(full_samples, args.split)
    save_samples(test_set, train_set, possible_columns, args.test_file, args.train_file)
    print("Sampling finished successfully.")


