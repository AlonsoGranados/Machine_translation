import torch
import random


SOS_token = 0
EOS_token = 1


def evaluate(encoder, decoder, sentence, max_length, eng_tokenizer, fr_tokenizer, device):
    with torch.no_grad():
        input_tensor = eng_tokenizer.encodeSequence(sentence, device)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(fr_tokenizer.token2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words

def evaluateRandomly(encoder, decoder, pairs, max_length, eng_tokenizer, fr_tokenizer, device, n = 10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words = evaluate(encoder, decoder, pair[0], max_length, eng_tokenizer, fr_tokenizer, device)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')