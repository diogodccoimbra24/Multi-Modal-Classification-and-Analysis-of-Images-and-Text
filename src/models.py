import keras
import tensorflow as tf
from tensorflow.keras import layers, Model


#Improved model (sequence - to sequence with fusion per timestep)
def build_caption_model(vocab_size, max_length = 22, embed_dim = 256, lstm_units = 256):


    T = max_length -1

    #Creating inputs
    #Precomputed ResNet50 feature vector
    img_input = layers.Input(shape = (2048, ), name = 'img_features')
    #caption tokens shifted for teacher forcing
    seq_input = layers.Input(shape=(T, ), dtype='int32', name = 'input_tokens')

    #Text branch
    txt = layers.Embedding(
        input_dim= vocab_size,
        output_dim= embed_dim,
        name = 'word_embedding'
    )(seq_input)
    txt = layers.LSTM(lstm_units, return_sequences= True, name= "text_lstm")(txt)

    #Image branch
    img = layers.Dense(
        lstm_units,
        activation= 'relu',
        name= 'img_proj'
    )(img_input)
    img = layers.RepeatVector(T, name= 'img_repeat')(img)

    #Fusion per timestep
    fused = layers.Concatenate(name= 'fusion_concat')([txt, img])
    fused = layers.LSTM(
        lstm_units,
        return_sequences= True,
        name= 'fusion_lstm'
    )(fused)

    #Predict next token at each timestep
    out = layers.TimeDistributed(
        layers.Dense(
            vocab_size,
            activation= 'softmax'
        ),
        name= 'vocab_softmax'
    )(fused)

    return Model(inputs=[img_input, seq_input], outputs=out, name="caption_model")


