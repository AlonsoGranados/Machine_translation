from __future__ import unicode_literals, print_function, division
import torch
import re

SOS_token = 0
EOS_token = 1

class Tokenizer:
    def __init__(self):
        self.word2token = {}
        self.word2count = {}
        self.token2word = {0: "SOS", 1: "EOS"}
        self.voc_size = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2token:
            self.word2token[word] = self.voc_size
            self.word2count[word] = 1
            self.token2word[self.voc_size] = word
            self.voc_size += 1
        else:
            self.word2count[word] += 1

    def encodeSequence(self, sentence, device):
        sequence = [self.word2token[word] for word in sentence.split(' ')]
        sequence.append(EOS_token)
        return torch.tensor(sequence, dtype=torch.long, device=device).view(-1, 1)


# Remove special characters and makes it lowercase
def cleanseString(s):
    s = re.sub(r'[^\w\s]', '', s).strip().lower()
    return s


def readData():

    # Read the file and split into lines
    lines = open('data/eng-fra.txt', encoding='utf-8').read().strip().split('\n')

    # Split into english and french, then preprocess data
    data = [[cleanseString(s) for s in l.split('\t')] for l in lines]

    return data

MAX_LENGTH = 8
MIN_LENGTH = 6

def removeLongSentences(pairs):
    return [pair for pair in pairs if ((MIN_LENGTH < len(pair[0].split(' ')) < MAX_LENGTH) and (MIN_LENGTH < len(pair[1].split(' ')) < MAX_LENGTH))]

def prepareData():
    # Get english2french data
    data = readData()
    data = removeLongSentences(data)

    eng_tokenizer = Tokenizer()
    fr_tokenizer = Tokenizer()

    for sentence in data:
        eng_tokenizer.addSentence(sentence[0])
        fr_tokenizer.addSentence(sentence[1])

    return eng_tokenizer, fr_tokenizer, data

