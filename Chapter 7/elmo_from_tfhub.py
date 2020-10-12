# Elmo: https://arxiv.org/abs/1802.05365
import tensorflow as tf
import tensorflow_hub as hub



tf.compat.v1.disable_eager_execution()
elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=False)

# "elmo" field: the weighted sum of the 3 layers, where the weights are trainable. This tensor has shape [batch_size, max_length, 1024]
embeddings = elmo(
    ["the cat is on the mat", "dogs are in the fog"],
    signature="default",
    as_dict=True)["elmo"]

print(embeddings.shape)