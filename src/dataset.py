
import numpy as np
from collections import defaultdict
import os
from image_processing import load_and_preprocess_image

#Function that builds a dictionary to match the image with the list of padded sequences
def build_image_to_captions(df, padded_sequences):

    image_to_captions = defaultdict(list)

    for i in range(len(df)):
        image_name = df.iloc[i]["image"]
        caption_seq = padded_sequences[i]
        image_to_captions[image_name].append(caption_seq)

    return image_to_captions

#Generator that give us the target word
def training_data_generator(
    image_to_captions,
    image_folder,
    pad_id=0,
    max_length=20
    ):

    for image_name, captions in image_to_captions.items():

        image_path = os.path.join(image_folder, image_name)
        image_tensor = load_and_preprocess_image(image_path)

        for caption in captions:

            #Find real caption length (before PAD)
            non_pad_positions = np.where(caption == pad_id)[0]
            cap_len = non_pad_positions[0] if len(non_pad_positions) > 0 else max_length

            #Teacher forcing: sliding window
            for t in range(1, cap_len):
                input_seq = caption.copy()
                input_seq[t:] = pad_id
                target_word = caption[t]

                yield (
                    image_tensor,
                    input_seq
                ), target_word

