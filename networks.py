import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class simpleEncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_prob):
        super(simpleEncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.RNN = nn.RNN(hidden_size, hidden_size, batch_first= True)

    def forward(self, input, hidden):
        output = self.embedding(input)
        output, hidden = self.RNN(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class simpleDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_prob):
        super(simpleDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input, hidden):
        input = input.view(-1,1)
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[:,0,:]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1 , self.hidden_size, device=device)

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_prob):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first= True)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input, hidden):
        output = self.dropout(self.embedding(input))
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_prob):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input, hidden):
        input = input.view(-1,1)
        output = self.dropout(self.embedding(input))
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[:,0,:]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1 , self.hidden_size, device=device)