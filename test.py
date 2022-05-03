# import nltk
#
#
#
# a = ['c', 'est', 'une', 'bonne', 'histoire']
# b = ['c', 'est', 'une', 'bonne', 'histoire']
#
# print(nltk.translate.bleu_score.sentence_bleu(a,b))

from nltk.translate.bleu_score import sentence_bleu
reference = ['this', 'is', 'a', 'test']
candidate = ['this', 'is', 'a', 'test']
score = sentence_bleu(reference, candidate)
print(score)