import numpy as np


class Embedding_layer:
    def __init__(self, vocab_size, embedding_dim):
        self.embeddings = np.random.randn(vocab_size, embedding_dim)*0.02
        self.inputs = None
    # Gradients
        self.grad_embeddings = np.zeros(self.embeddings.shape)

    def forward(self, inputs):
        self.inputs = inputs
        return self.embeddings[inputs]

    def backward(self, prev_grad):
        grad_embeddings = np.zeros(self.embeddings.shape)
        np.add.at(self.grad_embeddings, self.input, prev_grad)

        self.grad_embeddings += grad_embeddings
        return grad_embeddings

    def get_params(self):
        return [(self.embeddings, self.grad_embeddings)]


class Softmax:
    def __init__(self):
        self.softmax_out = None

    def forward(self, inputs):
        max_logit = np.max(inputs, axis=-1, keepdims=True)

        softmax_out = np.exp(
            inputs - max_logit) / np.sum(np.exp(inputs - max_logit), axis=-1, keepdims=True)
        self.softmax_out = softmax_out

        return softmax_out

    def backward(self, prev_grad):
        return prev_grad * self.softmax_out - self.softmax_out * np.sum(self.softmax_out * prev_grad, axis=-1, keepdims=True)


class CrossEntropyWithSoftmax:
    def __init__(self):
        self.lse = None
        self.logits_flat = None
        self.labels_flat = None

    def forward(self, logits, labels):
        assert len(logits.shape) == 3
        B, T, V = logits.shape

        logits_flat = np.reshape(logits, (B*T, V))
        labels_flat = np.reshape(labels, (B*T,))

        max_float = np.max(logits_flat, axis=-1, keepdims=True)
        diff = logits_flat - max_float

        lse = max_float + np.log(np.sum(np.exp(diff), axis=-1, keepdims=True))
        log_softmax = np.expand_dims(
            logits_flat[np.arrange(B*T), labels_flat], axis=-1) - lse

        self.lse = lse
        self.logits_flat = logits_flat
        self.labels_flat = labels_flat
        self.B = B
        self.T = T
        self.V = V

        return -np.sum(log_softmax)/logits_flat.shape[0]

    def backwards(self):
        N = self.logits_flat.shape[0]
        probs = np.exp(self.logits_flat - self.lse)

        grad_logits_flat = probs.copy()
        grad_logits_flat[np.arange[N], self.labels_flat] -= 1.0
        grad_logits_flat /= N

        return grad_logits_flat.reshape(self.B, self.T, self.V)
