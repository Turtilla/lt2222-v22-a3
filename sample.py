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

consonants_set = set(['b', 'c', 'd', 'f', 'g', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 's', 't', 'v', 'x', 'y', 'z', 'h', 'r', 't', 'w', 'y'])

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

def create_samples(sampled_lines, sample_number):
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
    cutoff = math.ceil(len(all_samples) * (test_percent / 100))
    test_set = all_samples[:cutoff]
    train_set = all_samples[cutoff:]
    return test_set, train_set

def save_samples(test_set, train_set, possible_columns, test_file, train_file):
    with open(test_file, 'wb') as testing_file:
        pickle.dump(test_set, testing_file)
    with open(train_file, 'wb') as training_file:
        pickle.dump((train_set, possible_columns), training_file)


if __name__ == "__main__":

    lines = sample_lines(args.input_file)
    full_samples = create_samples(lines, args.samples)
    possible_columns = create_columns(full_samples)
    test_set, train_set = split_samples(full_samples, args.split)
    save_samples(test_set, train_set, possible_columns, args.test_file, args.train_file)


