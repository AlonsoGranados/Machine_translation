import torch
import torch.nn as nn
from torch import optim
import random
import matplotlib.pyplot as plt

SOS_token = 0
EOS_token = 1


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length, device, teacher_forcing_ratio):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    loss = 0
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def trainIters(encoder, decoder, epochs, train_loader, val_loader, MAX_LENGTH, device, teacher_forcing_ratio, learning_rate=0.01):
    train_loss = []
    val_loss = []

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()

    for epoch in range(epochs):
        print(epoch)
        train_loss_per_epoch = 0
        val_loss_per_epoch = 0
        for j, data in enumerate(train_loader):
            loss = train(data[:][0], data[:][1], encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion, MAX_LENGTH, device, teacher_forcing_ratio)
            train_loss_per_epoch += loss

        train_loss.append(train_loss_per_epoch)
        for j, val_data in enumerate(val_loader):
            val_input = val_data[:][0]
            val_output = val_data[:][1]

            input_length = val_input.size(0)
            target_length = val_output.size(0)

            encoder_hidden = encoder.initHidden()
            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(
                    val_input[ei], encoder_hidden)

            decoder_input = torch.tensor([[SOS_token]], device=device)
            decoder_hidden = encoder_hidden

            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                val_loss_per_epoch += criterion(decoder_output, val_output[di])
                if decoder_input.item() == EOS_token:
                    break
            val_loss.append(val_loss_per_epoch)

    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.show()