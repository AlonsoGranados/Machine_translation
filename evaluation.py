import torch
import random


SOS_token = 0
EOS_token = 1


def evaluate(encoder, decoder, input_tensor, max_length, eng_tokenizer, fr_tokenizer, device):
    with torch.no_grad():
        # batch_size = input_tensor.size(0)
        input_length = input_tensor.size(0)
        encoder_hidden = encoder.initHidden()

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei].view(1,1),
                                                     encoder_hidden)
        # print(encoder_hidden)
        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        decoder_input = decoder_input.view(1,1)
        # decoder_input = torch.zeros(batch_size, dtype=torch.long, device=device).view(batch_size, 1)

        decoder_hidden = encoder_hidden
        # decoder_hidden = torch.rand(encoder_hidden.size(), device=device)
        decoded_words = []

        for di in range(max_length):

            decoder_output, decoder_hidden = decoder(
                decoder_input.view(1,1), decoder_hidden)
            topv, topi = decoder_output.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(fr_tokenizer.token2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words



def evaluateRandomly(encoder, decoder, test_loader, max_length, eng_tokenizer, fr_tokenizer, device, n = 50):
    for i in range(n):
        data = random.choice(test_loader)
        print('English: ', eng_tokenizer.decodeSequence(data[0]))
        print('French: ', fr_tokenizer.decodeSequence(data[1]))

        output_words = evaluate(encoder, decoder, data[0], max_length, eng_tokenizer, fr_tokenizer, device)
        output_sentence = ' '.join(output_words)
        print('Prediction: ', output_sentence)
        print('')