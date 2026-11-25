import numpy as np

from model import Embedding_layer, Softmax, CrossEntropyWithSoftmax

emb = Embedding_layer(vocab_size=4, embedding_dim=3)
emb.embeddings = np.array([
    [0.1, 0.2, 0.3],
    [1.0, 0.0, -1.0],
    [2.0, 2.0, 2.0],
    [0.5, -0.5, 0.0],
])

input_tokens = np.array([1, 3])
vectors = emb.forward(input_tokens)
print("1. Embedding vectors:\n", vectors)

sm = Softmax()
scores = np.array([3.0, 1.0, 0.2])
probs = sm.forward(scores)
print("\n2. softmax probs:", np.round(probs, 3))

loss_fn = CrossEntropyWithSoftmax()
logits = np.array([[[3.0, 1.0, 0.]]])
labels = np.array([[0]])

loss = loss_fn.forward(logits, labels)
grad_logits = loss_fn.backwards()
print("\n3. Loss:", round(float(loss), 3))
print(" Grad writ logits:", np.round(grad_logits, 3))
