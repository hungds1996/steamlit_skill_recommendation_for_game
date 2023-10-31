import tensorflow as tf

import numpy as np
import os
import time
import pandas as pd

from utils import *
from constants import all_skill

print("init_params")
# all_skill = get_skill_list() + ["end"]
ids_from_chars = lambda skills: [all_skill.index(s) for s in skills]
chars_from_ids = lambda ids: [all_skill[i] for i in ids]

vocab_size = len(all_skill)
embedding_dim = 128
rnn_units = 1024

class RNNModel(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, rnn_units):
    super().__init__(self)
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(rnn_units, return_sequences=True, return_state=True, dropout=0.1)
    self.dense = tf.keras.layers.Dense(vocab_size)

  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    x = self.embedding(x, training=training)
    if states is None:
      states = self.gru.get_initial_state(x)
    x, states = self.gru(x, initial_state=states, training=training)
    x = self.dense(x, training=training)

    if return_state:
      return x, states
    else:
      return x

class OneStep(tf.keras.Model):
    def __init__(self, model, chars_from_ids, ids_from_chars):
        super().__init__(self)
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars

        # Create a mask to prevent unwanted char
        chars_to_skip = ["end"]
        self.set_chars_to_skip(chars_to_skip)

    
    def set_chars_to_skip(self, skip_list=["end"]) -> tf.Tensor:
        skip_ids = tf.convert_to_tensor(self.ids_from_chars(skip_list), dtype=tf.int64)[:, None]

        sparse_mask = tf.SparseTensor(
            # Put a -inf at each bad index
            values=[-float('inf')]*len(skip_ids),
            indices=skip_ids,
            # match the shape to the vocabulary
            dense_shape = [len(all_skill)]
        )
        self.prediction_mask = tf.sparse.to_dense(tf.sparse.reorder(sparse_mask))
        

    # @tf.function
    def generate_one_step(self, inputs, states=None, temperature=1.0):
        # Convert to IDs
        input_ids = tf.convert_to_tensor([self.ids_from_chars(inputs)], dtype=tf.int64)

        # Run model
        # predicted_logits.shape is [batch, char, next_char_logits]
        predicted_logits, states = self.model(inputs=input_ids, states=states, return_state=True)

        # only take last prediction
        predicted_logits = predicted_logits[:, -1, :]
        
        if temperature > 0.0:
          predicted_logits = predicted_logits / temperature

        # apply the prediction mask: prevent "end" token
        predicted_logits = predicted_logits + self.prediction_mask

        # sample the output to generate token ids
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        # convert ids to chars
        predicted_chars = self.chars_from_ids(predicted_ids)

        return predicted_chars, states, predicted_logits


def get_model():
    model = RNNModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        rnn_units=rnn_units
    )

    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss)


    # Directory where the checkpoints will be saved
    model.built = True
    model.load_weights("./training_checkpoints/ckpt_150").expect_partial()

    one_step_model = OneStep(model, chars_from_ids, ids_from_chars)
    return one_step_model
  