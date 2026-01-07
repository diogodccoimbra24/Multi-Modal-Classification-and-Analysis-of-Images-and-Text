

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from src.text_processing import preprocess_captions, tokenize, vocabulary, padded_seq


CAPTIONS_PATH = "../data/captions/captions.txt"
IMAGE_FOLDER  = "../data/images/flickr8k_images/images"
MODEL_PATH    = "../models/caption_model_1.keras"
MAX_LENGTH    = 22


#Load model
model = tf.keras.models.load_model(MODEL_PATH)


#Building vocabulary
df = preprocess_captions(CAPTIONS_PATH)
df = tokenize(df)
_, word_to_idx = vocabulary(df)
idx_to_word = {i: w for w, i in word_to_idx.items()}

START_ID = word_to_idx["<START>"]
END_ID   = word_to_idx["<END>"]
PAD_ID   = word_to_idx["<PAD>"]


#ResNet50 feature extracting
resnet = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", pooling="avg")
resnet.trainable = False


def extract_single_feature(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img = tf.keras.utils.img_to_array(img)
    img = tf.expand_dims(img, axis=0)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    feat = resnet.predict(img, verbose=0)[0]  # (2048,)
    return feat.astype(np.float32)

#Greedy decoding to predict the next token step by step
def generate_caption(image_path, max_length=MAX_LENGTH):

    T = max_length - 1

    img_feat = extract_single_feature(image_path)[None, :]

    seq = [START_ID]

    for _ in range(T):
        #Build fixed-length input tokens
        inp = seq + [PAD_ID] * (T - len(seq))
        inp = np.array(inp[:T], dtype=np.int32)[None, :]

        preds = model.predict([img_feat, inp], verbose=0)[0]

        step = len(seq) - 1
        next_id = int(np.argmax(preds[step]))

        if next_id == END_ID:
            break

        seq.append(next_id)

        if len(seq) >= max_length:
            break

    #Convert to words
    words = []
    for tid in seq:
        if tid in (START_ID, PAD_ID):
            continue
        if tid == END_ID:
            break
        words.append(idx_to_word.get(tid, "<UNK>"))

    return " ".join(words)


def show_image_with_caption(image_name):
    image_path = os.path.join(IMAGE_FOLDER, image_name)
    caption = generate_caption(image_path)

    img = mpimg.imread(image_path)
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.title(caption, fontsize=10)
    plt.show()

#To try one image
if __name__ == "__main__":
    test_image = df["image"].iloc[2004]
    show_image_with_caption(test_image)
