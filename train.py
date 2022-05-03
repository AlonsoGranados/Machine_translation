import torch
import torch.nn as nn
from torch import optim
import random
import matplotlib.pyplot as plt

SOS_token = 0
EOS_token = 1


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length, device, teacher_forcing_ratio):
    # encoder_hidden = encoder.initHidden()



    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    batch_size = input_tensor.size(0)
    # input_length = input_tensor.size(0)
    # target_length = target_tensor.size(0)
    # Modify for batch
    input_length = input_tensor.size(1)
    target_length = target_tensor.size(1)

    encoder_hidden = torch.zeros(1, batch_size, encoder.hidden_size, device=device)

    loss = 0
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[:,ei], encoder_hidden)

    # decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_input = torch.zeros(batch_size, dtype = torch.long, device =device).view(1,batch_size)
    decoder_hidden = encoder_hidden

    # Feed target during training process
    if (random.random() < teacher_forcing_ratio):
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)

            # print("Predicted")
            # print(decoder_output.shape)
            # print("Expected")
            # print(target_tensor[:,di].shape)
            # print(target_tensor.shape)

            loss += criterion(decoder_output.view(batch_size,decoder.output_size), target_tensor[:,di].view(-1))
            decoder_input = target_tensor[:,di]  # Teacher forcing

    else:
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)

            # print(topi.size())

            decoder_input = topi.squeeze().detach()  # detach from history as input
            decoder_input = decoder_input.view(batch_size,1)
            # print('TESTING')
            # print(decoder_input.shape)
            # print(decoder_output.shape)
            # print(target_tensor[:,di].shape)
            loss += criterion(decoder_output.view(batch_size, decoder.output_size), target_tensor[:,di].view(-1))

            # print('Comparison')
            # print(decoder_input)
            # print(target_tensor[:,di])
            # if decoder_input.item() == EOS_token:
            #     break
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def trainIters(encoder, decoder, epochs, train_loader, val_loader, MAX_LENGTH, device, teacher_forcing_ratio, learning_rate=0.01):
    train_loss = []
    val_loss = []

    # encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    # decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    encoder_optimizer = optim.Adam(encoder.parameters())
    decoder_optimizer = optim.Adam(decoder.parameters())


    criterion = nn.NLLLoss()
    # criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        print(epoch)
        train_loss_per_epoch = 0
        val_loss_per_epoch = 0
        for j, data in enumerate(train_loader):
            # print(data[0].shape)
            # print(len(data))
            loss = train(data[0], data[1], encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion, MAX_LENGTH, device, teacher_forcing_ratio)
            train_loss_per_epoch += loss

        train_loss.append(train_loss_per_epoch/len(train_loader))
        # Validation step
        for j, val_data in enumerate(val_loader):
            val_input = val_data[0]
            val_output = val_data[1]

            input_length = val_input.size(1)
            target_length = val_output.size(1)
            batch_size = val_input.size(0)

            # encoder_hidden = encoder.initHidden()
            encoder_hidden = torch.zeros(1, batch_size, encoder.hidden_size, device=device)
            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(
                    val_input[:,ei], encoder_hidden)

            # decoder_input = torch.tensor([[SOS_token]], device=device)
            decoder_input = torch.zeros(batch_size, dtype=torch.long, device=device).view(batch_size, 1)
            decoder_hidden = encoder_hidden

            for di in range(target_length):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()
                decoder_input = decoder_input.view(batch_size, 1)

                val_loss_per_epoch += criterion(decoder_output.view(batch_size, decoder.output_size, 1), val_output[:,di])


                # if decoder_input.item() == EOS_token:
                #     break
        val_loss.append(val_loss_per_epoch /(len(val_loader)*MAX_LENGTH))
        if((epoch + 1)% 5 == 0):
            plt.title("RNN model Training Phase")
            plt.plot(train_loss, label = 'Cumulative Training Loss')
            plt.plot(val_loss, label = 'Cumulative Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()