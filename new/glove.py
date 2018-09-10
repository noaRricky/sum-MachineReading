import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np


class Glove(nn.Module):

    def __init__(self, context_size, embed_size, vocab_size, xmax=2, alpha=0.75):
        super(Glove, self).__init__()

        self.context_size = context_size
        self.embed_size = embed_size
        self.xmax = xmax
        self.alpha = alpha
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.embedding_bias = nn.Parameter(torch.zeros(embed_size))
        self.linear = nn.Linear(embed_size, vocab_size)

    def forward(self, tokens):
        """embed the words in the sentence

        Arguments:
            tokens {tensor} -- shape (*, token), where * reference to the data shape
        """
        return self.embedding(tokens) + self.embedding_bias

    def fit(self, corpus):
        """fit the model with corpus

        Arguments:
            corpus {list} -- index word store in corpus, corpus -> sentences -> word
        """
        comat: np.array = self._get_comatrix(corpus)

    def _get_comatrix(self, corpus) -> np.array:
        """compute the co-occurence probabilities of corpus
        
        Arguments:
            corpus {list} -- index word store in corpus, corpus -> sentences -> word
        
        Returns:
            [np.array] -- co-occrrence matrix
        """

        vocab_size, context_size = self.vocab_size, self.context_size

        # construct co-occurence matrix
        ctvect = np.zeros(vocab_size)
        comat = np.zeros((vocab_size, vocab_size))
        for sentence in corpus:
            sent_len = len(sentence)
            for i, word in enumerate(sentence):
                if i - context_size > 0:
                    ctvect[word] += context_size
                else:
                    ctvect[word] += i
                if i + context_size < sent_len:
                    ctvect[word] += context_size + 1
                else:
                    ctvect[word] += sent_len - i

        for sentence in corpus:
            sent_len = len(sentence)
            for i, center_word in enumerate(sentence):
                for j in range(1, context_size + 1):
                    if i - j > 0:
                        context_word = sentence[i - j]
                        comat[center_word, context_word] += 1
                    if i + j < sent_len:
                        context_word = sentence[i + j]
                        comat[center_word, context_word] += 1
        comat /= ctvect[:, np.newaxis]
        comat[comat == np.nan] = 0
        return comat


if __name__ == '__main__':
    context_size = 2
    embed_size = 124
    vocab_size = 6
    glove = Glove(context_size, embed_size, vocab_size)
    corpus = [[0, 1, 2, 3, 4], [2, 3, 5, 1, 2, 1, 2]]
    glove.fit(corpus)
