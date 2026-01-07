
import numpy as np
import tensorflow as tf
import os

#Precomputes ResNet50 features for each image
def extract_image_features(image_folder, image_names):
    resnet = tf.keras.applications.ResNet50(
        include_top= False,
        weights= 'imagenet',
        pooling= 'avg'
    )
    #To freeze
    resnet.trainable = False

    features = {}

    for img_name in image_names:
        img_path = os.path.join(image_folder, img_name)
        if not os.path.exists(img_path):
            continue

        #Load + preprocess image
        img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
        img = tf.keras.utils.img_to_array(img)
        img = tf.expand_dims(img, axis=0)
        img = tf.keras.applications.resnet50.preprocess_input(img)

        #Extract features (2048,)
        feat = resnet.predict(img, verbose=0)[0]
        features[img_name] = feat.astype(np.float32)
    return features

#Function that builds arrays for sequence to sequence teacher forcing
def build_sequence_dataset(df, padded_sequences, image_features):

    X_img, X_in, Y_out = [], [], []

    for i in range(len(df)):
        img_name = df.iloc[i]["image"]
        if img_name not in image_features:
            continue

        cap = padded_sequences[i]
        X_img.append(image_features[img_name])
        X_in.append(cap[:-1])
        Y_out.append(cap[1:])

    return(
        np.asarray(X_img, dtype=np.float32),
        np.asarray(X_in, dtype=np.int32),
        np.asarray(Y_out, dtype=np.int32),
    )
