

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Embedding, LSTM

#Text encoder model
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


#Image encoder model
def build_image_encoder():
    model = tf.keras.applications.ResNet50(
        include_top = False,
        weights = 'imagenet',
        input_tensor = None,
        input_shape = (224, 224, 3),
        pooling = 'avg',
        classes = None,
        classifier_activation = None
    )

    #Freezing the model
    model.trainable = False
    return model

build_image_encoder().summary()



