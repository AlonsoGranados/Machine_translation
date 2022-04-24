from networks import EncoderRNN
from networks import DecoderRNN
from read_data import prepareData
from train import trainIters
from sklearn.model_selection import train_test_split

import torch

# Parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 8

eng_tokenizer, fr_tokenizer, data = prepareData()
data = data[:5000]

# Encode data
data = [(eng_tokenizer.encodeSequence(sentence[0],device), fr_tokenizer.encodeSequence(sentence[1],device)) for sentence in data]

train_data, test_data = train_test_split(data, test_size=0.10, random_state=42)
train_data, validation_data = train_test_split(train_data, test_size=0.10, random_state=42)

# Parameters
teacher_forcing_ratio = 0.5
hidden_size = 256
encoder1 = EncoderRNN(eng_tokenizer.voc_size, hidden_size).to(device)
decoder1 = DecoderRNN(hidden_size, fr_tokenizer.voc_size).to(device)

# from torch.utils.data import DataLoader
# train_loader = DataLoader(train_data,shuffle=True,batch_size=256)

trainIters(encoder1, decoder1, 10, train_data, validation_data, MAX_LENGTH, device, teacher_forcing_ratio)
#
# evaluateRandomly(encoder1, decoder1, data, MAX_LENGTH, eng_tokenizer, fr_tokenizer, device)