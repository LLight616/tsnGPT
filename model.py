import numpy as np


class Embedding_layer:
    def __init__(self, vocab_size, embedding_dim):
        self.embeddings = np.random.randn(vocab_size, embedding_dim)*0.02
        self.input = None
    # Gradients
        self.grad_embeddings = np.zeros(self.embeddings.shape)

    def forward(self, input):
        self.input = input
        return self.embeddings[input]
