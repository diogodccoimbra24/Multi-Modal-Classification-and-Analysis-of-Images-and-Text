

from text_processing import preprocess_captions, tokenize, vocabulary, padded_seq
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Embedding, LSTM


def build_text_encoder(vocab_size, max_length=20):

    #Default dimension for small / medium datasets (flickr8k)
    embedding_dim = 128

    #Building the model
    model = models.Sequential([
        Embedding(input_dim = vocab_size,
                  output_dim = embedding_dim,
                  input_length = max_length),
        LSTM(256) #Standard default
    ])

    model.build(input_shape=(None, max_length))
    return model

