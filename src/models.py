import keras
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Embedding, LSTM
from keras.models import Model

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

#fusion model (between text encoder and image encoder)
def build_fusion_model(vocab_size):

    # Using directly the encoders
    image_encoder = build_image_encoder()
    text_encoder = build_text_encoder(vocab_size)

    #Creating inputs for the encoders
    image_input = keras.Input(shape = (224, 224, 3), name = 'image_input')
    text_input = keras.Input(shape=(20, ), dtype='int32')


    #Calling the functions of the encoders with the inputs
    image_feature = image_encoder(image_input)
    text_feature = text_encoder(text_input)

    #Concatenation the outputs from the encoders
    fused = keras.layers.concatenate([image_feature,text_feature], -1)


    #Hidden layer
    x = keras.layers.Dense(
        units = 256,
        activation = 'relu'
        )(fused)

    #Output layer
    outputs = keras.layers.Dense(
        units = vocab_size,
        activation = 'softmax'
        )(x)



    #returnin the model with the inputs and output
    return Model(inputs = [image_input, text_input], outputs = outputs)

fusion_model = build_fusion_model(vocab_size= 8829)
fusion_model.summary()


