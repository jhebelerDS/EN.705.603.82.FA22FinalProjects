"""This is the module for the data classes and functions. Should only need to be run once
if the dataset has been downloaded and put into /inputs."""

import random
import re
from tqdm import tqdm


class Language:
    PAD_idx = 0 # padding
    SOS_idx = 1 # start of sentence
    EOS_idx = 2 # end of sentence
    UNK_idx = 3 # unknown word

    def __init__(self):
        self.word2count = {}
        self.word2index = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.index2word = {v: k for k, v in self.word2index.items()}
        self.n_words = 4
        self.max_length = 0

    def add_sentence(self, sentence):
        words = self.sentence_to_words(sentence)

        if len(words) > self.max_length:
            self.max_length = len(words)

        for word in words:
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    @classmethod
    def create_vocabs(cls, pairs, src_kwargs={}, trg_kwargs={}):

        src_lang = cls(**src_kwargs)
        trg_lang = cls(**trg_kwargs)

        for src, trg in tqdm(pairs, desc="creating vocabs"):
            src_lang.add_sentence(src)
            trg_lang.add_sentence(trg)

        return src_lang, trg_lang


class PolynomialLanguage(Language):
    def sentence_to_words(self, sentence):
        return re.findall(r"sin|cos|tan|\d|\w|\(|\)|\+|-|\*+", sentence.strip().lower())

    def words_to_sentence(self, words):
        return "".join(words)

    @staticmethod
    def load_pairs(filepath, reverse=False):
        with open(filepath) as fi:
            pairs = [line.strip().split("=") for line in fi]

        if reverse:
            pairs = [(b, a) for a, b in pairs]

        return pairs

def train_test_split(pairs: list, ratio: float) -> tuple:
    """This function splits the data into train and test sets.
    Args: pairs: A list of tuples of the form (factors, expansions)
          ratio: the ratio of the train set to the test set
    Returns: train_pairs, test_pairs"""

    random.shuffle(pairs) # shuffle the data
    split = int(ratio * len(pairs)) # calculate the split point
    train, test = pairs[:split], pairs[split:] # split the data at split point
    return train, test # ruturn tuple: train, test


def load_file(file_path: str) -> tuple:
    """ A helper functions that loads the file into a tuple of strings
    Args: file_path
    Returns: factors: inputs to the model, expansions: group truth"""

    data = open(file_path, "r").readlines()
    factors, expansions = zip(*[line.strip().split("=") for line in data])
    return factors, expansions


if __name__ == "__main__":
    print("Reading data...")

    # prompt for ratio of data to use for training
    split_ratio = float(input("Enter the ratio of data to use for training (float): "))

    with open('inputs/data.txt') as file_input:
        element_per_line = file_input.read().splitlines() # read the data and split into lines
        # split the data into train and test sets
        train_pairs, test_pairs = train_test_split(element_per_line, split_ratio)

    with open('inputs/train_set.txt', "w") as file_output: # write the train set to file
        file_output.write("\n".join(train_pairs) + "\n")

    with open('inputs/test_set.txt', "w") as file_output: # write the test set to file
        file_output.write("\n".join(test_pairs) + "\n")
