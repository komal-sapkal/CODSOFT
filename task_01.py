import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoModelForSequenceClassification
sns.set()

train_data_file = "Genre Classification Dataset/train_data.txt"
test_data_file = "Genre Classification Dataset/test_data_solution.txt"

with open(train_data_file, 'r') as file:
    print(f"First line (header) looks like this:\n\n{file.readline()}")
    print(f"Each data point looks like this:\n\n{file.readline()}")
NUM_WORDS = 5000
EMBEDDING_DIM = 16
MAXLEN = 250
PADDING = 'post'
OOV_TOKEN = "<OOV>"
def preprocess_text(sentence):
    """Cleans and prepares text data for model input.

    Args:
        sentence: The raw text sentence to be processed.

    Returns:
        The cleaned and preprocessed sentence.
    """
    stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]
    
    sentence = sentence.lower()

    words = sentence.split()
    no_words = [w for w in words if w not in stopwords]
    sentence = " ".join(no_words)

    return sentence


def extract_data_from_source(filename):
    sentences = []
    labels = []

    with open(filename, 'r') as file:
        for line in file:
            values = line.strip().split(':::')
            labels.append(values[2])
            sentence = values[3]
            sentence = preprocess_text(sentence)
            sentences.append(sentence)

    return sentences, labels
sentences, labels = extract_data_from_source(train_data_file)

print(f"There are {len(sentences)} sentences in the dataset.\n")
print(f"First sentence has {len(sentences[0].split())} words (after removing stopwords).\n")
print(f"There are {len(labels)} labels in the dataset.\n")
print(f"The first 5 labels are {labels[:5]}")
print(sentences[0])
test_sentences, test_labels = extract_data_from_source(test_data_file)

print(f"There are {len(test_sentences)} sentences in the dataset.\n")
print(f"First sentence has {len(test_sentences[0].split())} words (after removing stopwords).\n")
print(f"There are {len(test_labels)} labels in the dataset.\n")
print(f"The first 5 labels are {test_labels[:5]}")
print(f"There are {len(sentences)} sentences for training.\n")
print(f"There are {len(labels)} labels for training.\n")
print(f"There are {len(test_sentences)} sentences for validation.\n")
print(f"There are {len(test_labels)} labels for validation.")
def fit_tokenizer(train_sentences, num_words, oov_token):
    tokenizer = Tokenizer(oov_token = oov_token, num_words = num_words)
    tokenizer.fit_on_texts(train_sentences)
    
    return tokenizer
tokenizer = fit_tokenizer(sentences, NUM_WORDS, OOV_TOKEN)
word_index = tokenizer.word_index

print(f"Vocabulary contains {len(word_index)} words\n")
print("<OOV> token included in vocabulary" if "<OOV>" in word_index else "<OOV> token NOT included in vocabulary")
def seq_and_pad(sentences, tokenizer, padding, maxlen):
    sequences = tokenizer.texts_to_sequences(sentences)
    padded_sequences = pad_sequences(sequences, maxlen = maxlen, padding = padding)

    return padded_sequences
prepared_training_sequences = seq_and_pad(sentences, tokenizer, PADDING, MAXLEN)
val_padded_seq = seq_and_pad(test_sentences, tokenizer, PADDING, MAXLEN)

print(f"Padded training sequences have shape: {prepared_training_sequences.shape}\n")
print(f"Padded validation sequences have shape: {val_padded_seq.shape}")
def remove_special_character(word, special_char='-'):
    return word.replace(special_char, '')

def replace_words_with_special_character(words_list, special_char='-'):
    new_words_list = []
    for word in words_list:
        if special_char in word:
            new_word = remove_special_character(word, special_char)
            new_words_list.append(new_word)
        else:
            new_words_list.append(word)
    return new_words_list

def count_unique_elements(input_list):
    unique_elements = set(input_list)
    return len(unique_elements)
labels = replace_words_with_special_character(labels)
test_labels = replace_words_with_special_character(test_labels)
no = count_unique_elements(labels)
no
def flatten_list(input_list):
    output_list = []
    for item in input_list:
        if isinstance(item, list):
            output_list.extend(flatten_list(item))
        else:
            output_list.append(item)
    return output_list
def tokenize_labels(labels):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(labels)
    tokenized_labels = tokenizer.texts_to_sequences(labels)
    tokenized_labels = np.array(tokenized_labels) - 1

    return tokenized_labels
train_label_seq = tokenize_labels(labels)
val_label_seq = tokenize_labels(test_labels)
print(f"First 5 labels of the training set should look like this:\n{train_label_seq[:5]}\n")
print(f"First 5 labels of the validation set should look like this:\n{val_label_seq[:5]}\n")
print(f"Tokenized labels of the training set have shape: {train_label_seq.shape}\n")
print(f"Tokenized labels of the validation set have shape: {val_label_seq.shape}\n")

def create_model(num_words, embedding_dim, maxlen):
    tf.random.set_seed(123)
    
    model = tf.keras.Sequential([ 
        tf.keras.layers.Embedding(num_words, embedding_dim, input_length=maxlen),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(96, activation='relu'),
        tf.keras.layers.Dense(27, activation='softmax'),
    ])
    
    model.compile(loss="sparse_categorical_crossentropy",
                optimizer="adam",
                metrics=['accuracy']) 

    return model

model = create_model(NUM_WORDS, EMBEDDING_DIM, MAXLEN)

history = model.fit(prepared_training_sequences, train_label_seq, epochs=2, validation_data=(val_padded_seq, val_label_seq)) #change epoch="" to the number of iterations as per need

def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history[f'val_{metric}'])
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, f'val_{metric}'])
    plt.show()
    
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")