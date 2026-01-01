

from os import truncate
import pandas as pd
from keras import Layer
from tensorflow.python.keras import Sequential
import string

#Path to the captions.txt file
captions_path = "../data/captions/captions.txt"


def preprocess_captions(captions_path):

    #Reading the file
    df = pd.read_csv(captions_path)

    # Removing the upper cases
    df['lower_caption'] = df['caption'].str.lower()

    # Function to remove the punctuation
    def remove_punctuation(lower_caption):
        for char in string.punctuation:
            lower_caption = lower_caption.replace(char, '')
        return lower_caption

    df['lower_caption'] = df['lower_caption'].apply(remove_punctuation)

    # Comparing the new column with the old column
    return df.sample(10)


print(preprocess_captions(captions_path))
