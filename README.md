# Machine Translation English2French: A CS539 NLP Project

Project structure: 

Run main file to start training network, and evaluate over the Testset.

Networks.py: This file includes both network architectures RNN and GRU.

read_data.py: This file reads the text file and encodes sentences into sequences of tokens.

train.py: This file includes the logic to get mini batches, forward and backward pass, and compute validation loss.

evaluation.py: This file executes a forward pass for a sample.

Dataset can be downloaded at:

https://www.manythings.org/anki/


Credits to Sean Robertson for his pytorch tutorial that helped me frame the training step:

https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
