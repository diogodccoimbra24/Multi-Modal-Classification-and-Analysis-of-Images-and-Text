
import tensorflow as tf
from src.dataset import training_data_generator, build_image_to_captions
from src.text_processing import preprocess_captions, tokenize, vocabulary, padded_seq



CAPTIONS_PATH = "../data/captions/captions.txt"
MAX_LENGTH = 22

df = preprocess_captions(CAPTIONS_PATH)
df = tokenize(df)

sequences, word_to_idx = vocabulary(df)
padded_sequences = padded_seq(sequences, max_length=MAX_LENGTH)

print("df shape:", df.shape)
print("padded_sequences shape:", padded_sequences.shape)
print("first seq (ids):", padded_sequences[0][:10])



MAX_LENGTH = 22
BATCH_SIZE = 32

image_to_captions = build_image_to_captions(df, padded_sequences)


image_folder = "../data/images/flickr8k_images/images"

dataset = tf.data.Dataset.from_generator(
    lambda: training_data_generator(
        image_to_captions,
        image_folder,
        pad_id=0,
        max_length=MAX_LENGTH
    ),
    output_signature=(
        (
            tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(MAX_LENGTH,), dtype=tf.int32),
        ),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
)

dataset = dataset.batch(BATCH_SIZE)



from src.models import fusion_model

fusion_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


fusion_model.fit(dataset, epochs=2)