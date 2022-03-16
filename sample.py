import argparse
import gzip
import random

parser = argparse.ArgumentParser()
parser.add_argument("file", help="the file you want to make the samples out of (.txt or .gzip)")
parser.add_argument("samples", help="the number of samples that you want to produce", type=int)
parser.add_argument("split", help="the percentage of the samples that will form the test set (e.g. entering 20 here will produce a 20/80 test/train split)", type=int)

args = parser.parse_args()

consonants_set = set(['b', 'c', 'd', 'f', 'g', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 's', 't', 'v', 'x', 'y', 'z', 'h', 'r', 't', 'w', 'y'])

def sample_lines(filename, lines):
    if filename.endswith(".gz"):
        unzipped_file = gzip.open(filename, "rb")
        contents = unzipped_file.readlines()
        random_slice = random.randint(0, (len(contents)-lines))
        lines = contents[random_slice:random_slice+lines]
        sampled_lines = []
        for line in lines:
            decoded_line = line.decode('utf8').strip().lower()
            sampled_lines.append(decoded_line)
        unzipped_file.close()
        return sampled_lines

    elif filename.endswith(".txt"):
        opened_file = open(filename, "r")
        contents = opened_file.readlines()
        random_slice = random.randint(0, (len(contents)-lines))
        lines = contents[random_slice:random_slice+lines]
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
    sample_counter = 0
    all_samples = []
    for line in sampled_lines:
        for i in range(0, len(line)-5):
            four_characters = (line[i]+"_1", line[i+1]+"_2", line[i+2]+"_3", line[i+3]+"_4")
            closest_consonant = find_closest_consonant(line[i+4:])
            if closest_consonant != False:
                full_sample = (four_characters, closest_consonant)
                all_samples.append(full_sample)

    return all_samples
                
                    


if __name__ == "__main__":

    lines = sample_lines(args.file, args.samples)
    print(lines)
    full_samples = create_samples(lines, args.samples)
    print(full_samples)