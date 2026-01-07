
import tensorflow as tf
from src.dataset import extract_image_features, build_sequence_dataset
from src.text_processing import preprocess_captions, tokenize, vocabulary, padded_seq
from src.models import build_caption_model


CAPTIONS_PATH = "../data/captions/captions.txt"
IMAGE_FOLDER = "../data/images/flickr8k_images/images"

MAX_LENGTH = 22
BATCH_SIZE = 64
EPOCHS = 10

#Text processing
df = preprocess_captions(CAPTIONS_PATH)
df = tokenize(df)

sequences, word_to_idx = vocabulary(df)
padded_sequences = padded_seq(sequences, max_length=MAX_LENGTH)

#Precompute CNN features (2048 per image)
image_features = extract_image_features(
    IMAGE_FOLDER,
    df["image"].unique()
)

#Build seq2seq dataset arrays
X_img, X_in, Y_out = build_sequence_dataset(df, padded_sequences, image_features)

print("X_img:", X_img.shape)
print("X_in :", X_in.shape)
print("Y_out:", Y_out.shape)

#Tf.data pipeline
ds = tf.data.Dataset.from_tensor_slices(((X_img, X_in), Y_out))
ds = ds.shuffle(5000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

#Model
model = build_caption_model(
    vocab_size= len(word_to_idx),
    max_length= MAX_LENGTH
)
model.compile(
    optimizer= 'adam',
    loss= 'sparse_categorical_crossentropy',
    matrics= ['accuracy']
)

#Train te model
history = model.fit(ds, epochs= EPOCHS, steps_per_epoch= 1000)


#Save model
model.save("../models/caption_model.keras")

#Save history for plots later
import json
with open("../models/history.json", "w") as f:
    json.dump(history.history, f)