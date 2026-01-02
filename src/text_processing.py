
import pandas as pd
import string
from nltk.tokenize import word_tokenize
from collections import Counter
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Path to the captions.txt file
captions_path = "../data/captions/captions.txt"


def preprocess_captions(captions_path):

    #Reading the file
    df = pd.read_csv(captions_path)

    #Removing the upper cases
    df['lower_caption'] = df['caption'].str.lower()

    #Function to remove the punctuation
    def remove_punctuation(lower_caption):
        for char in string.punctuation:
            lower_caption = lower_caption.replace(char, '')
        return lower_caption

    df['lower_caption'] = df['lower_caption'].apply(remove_punctuation)

    return df


def tokenize(df):

    #Creating a new column for tokens for each caption
    df['tokens'] = df['lower_caption'].apply(lambda x: word_tokenize(x))

    return df


def vocabulary(df):
    # Counting all the words
    word_counter = Counter()

    for i in df['tokens']:
        word_counter.update(i)

    #Defining special tokens
    #To fill empty positions
    PAD_TOKEN = "<PAD>"
    #When a word is not in the vocabulary
    UNK_TOKEN = "<UNK>"

    #Creating a vocabulary
    word_to_idx = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1
    }

    for word in word_counter:
        #Auto increments the ID generator
        word_to_idx[word] = len(word_to_idx)

    #Creating a variable for unk token index
    UNK_IDX = word_to_idx[UNK_TOKEN]

    sequence = []

    #For loop to check if the word is in the tokens dataframe; if it is just adds the index; if it's not it will add the unk index
    for tokens in df['tokens']:
        seq = []
        for word in tokens:
            if word in word_to_idx:
                seq.append(word_to_idx[word])
            else:
                seq.append(UNK_IDX)
        #Appending to the empty sequence list
        sequence.append(seq)

    return sequence, word_to_idx


def padded_seq(sequence, max_length = 20):
    # Truncating sequences > 20
    # Padding sequences < 20
    max_length = 20

    #Sequences padded and truncated
    padded_sequences = pad_sequences(
        sequence,
        maxlen=max_length,
        # Padding at the end
        padding='post',
        # Truncating at the end
        truncating='post'
    )

    return padded_sequences




