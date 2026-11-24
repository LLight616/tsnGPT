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

    def backward(self, prev_grad):
        grad_embeddings = np.zeros(self.embeddings.shape)
        np.add.at(self.grad_embeddings, self.input, prev_grad)

        self.grad_embeddings += grad_embeddings
        return grad_embeddings
