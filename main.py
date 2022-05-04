from networks import simpleEncoderRNN
from networks import simpleDecoderRNN
from networks import EncoderRNN
from networks import DecoderRNN
from read_data import prepareData
from train import trainIters
from sklearn.model_selection import train_test_split
from evaluation import evaluate


import torch

# Parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 6

eng_tokenizer, fr_tokenizer, data = prepareData()

# Encode data
data = [(eng_tokenizer.encodeSequence(sentence[0],device), fr_tokenizer.encodeSequence(sentence[1],device)) for sentence in data]


train_data, test_data = train_test_split(data, test_size=0.10, random_state=42)
train_data, validation_data = train_test_split(train_data, test_size=0.10, random_state=42)

# Parameters
teacher_forcing_ratio = 0.5
dropout_rate = 0.25
hidden_size = 256
# encoder = simpleEncoderRNN(eng_tokenizer.voc_size, hidden_size,dropout_prob=0.5).to(device)
# decoder = simpleDecoderRNN(hidden_size, fr_tokenizer.voc_size, dropout_prob=0.5).to(device)
encoder = EncoderRNN(eng_tokenizer.voc_size, hidden_size, dropout_prob= dropout_rate).to(device)
decoder = DecoderRNN(hidden_size, fr_tokenizer.voc_size, dropout_prob= dropout_rate).to(device)


from torch.utils.data import DataLoader
train_loader = DataLoader(train_data,shuffle=True,batch_size=32)
val_loader = DataLoader(validation_data,shuffle=True,batch_size=32)
test_loader = DataLoader(test_data,shuffle=True,batch_size=32)


trainIters(encoder, decoder, 20, train_loader, val_loader, MAX_LENGTH, device, teacher_forcing_ratio)

encoder.dropout = torch.nn.Dropout(0)
decoder.dropout = torch.nn.Dropout(0)


import nltk
average_blue_score = 0
one_gram = 0
bi_gram = 0
tri_gram = 0
tetra_gram = 0
errors = 0
for i in range(len(test_data)):
    # print(fr_tokenizer.decodeSequence(test_data[i][1]).split())
    # print(evaluate(encoder,decoder,test_data[i][0],MAX_LENGTH,eng_tokenizer, fr_tokenizer, device))
    label = fr_tokenizer.decodeSequence(test_data[i][1]).split()[:-1]
    pred = evaluate(encoder,decoder,test_data[i][0],MAX_LENGTH,eng_tokenizer, fr_tokenizer, device)[:-1]

    average_blue_score += nltk.translate.bleu_score.sentence_bleu([label], pred, weights=(0.25,0.25,0.25,0.25))
    one_gram += nltk.translate.bleu_score.sentence_bleu([label], pred, weights=(1, 0, 0, 0))
    bi_gram += nltk.translate.bleu_score.sentence_bleu([label], pred, weights=(0, 1, 0, 0))
    b = nltk.translate.bleu_score.sentence_bleu([label], pred, weights=(0, 1, 0, 0))
    if(b < 0.4):
        errors += 1
        print(errors, b)
        print(label)
        print(pred)
    tri_gram += nltk.translate.bleu_score.sentence_bleu([label], pred, weights=(0, 0, 1, 0))
    tetra_gram += nltk.translate.bleu_score.sentence_bleu( [label], pred, weights=(0, 0, 0, 1))
    # print(nltk.translate.bleu_score.sentence_bleu(
    #     [fr_tokenizer.decodeSequence(test_data[i][1]).split()[:-1]], evaluate(encoder,decoder,test_data[i][0],MAX_LENGTH,eng_tokenizer, fr_tokenizer, device)[:-1],weights=(0.5,0.5,0,0)))
print((one_gram + bi_gram + tri_gram + tetra_gram)/ (4*len(test_data)))
print(one_gram/ len(test_data))
print(bi_gram/ len(test_data))
print(tri_gram/ len(test_data))
print(tetra_gram/ len(test_data))
print(errors)
