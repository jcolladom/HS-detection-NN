# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import re, random
import liwc
import tensorflow as tf

from gensim.models import KeyedVectors
from statistics import mean, median
from collections import Counter
from tensorflow import keras
from tensorflow.keras import preprocessing
from tensorflow.keras import layers, initializers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix

class SpanishLIWC:
  def __init__(self, path):
    self.path = path + 'Spanish_LIWC.dic'

  # Tokenizer function
  def tokenize(self, text):
    # you may want to use a smarter tokenizer
    for match in re.finditer(r'\w+', text, re.UNICODE):
      yield match.group(0)

  # Emotions counter for the LIWC lexicon
  # Returns: a dictionary with emotions as keys and its frequency on each tweet as value
  def count_emotions(self, dataset, parse):
    EmoPos = []
    EmoNeg = []
    Enfado = []
    Triste = []
    Ansiedad = []
    emodict = {}

    for tweet in dataset: 
      tokens = self.tokenize(tweet)
      tokens_count = Counter(category for token in tokens for category in parse(token))
      EmoPos.append(tokens_count['EmoPos'])
      EmoNeg.append(tokens_count['EmoNeg'])
      Enfado.append(tokens_count['Enfado'])
      Triste.append(tokens_count['Triste'])
      Ansiedad.append(tokens_count['Ansiedad'])

    emodict['EmoPos'] = EmoPos
    emodict['EmoNeg'] = EmoNeg
    emodict['Enfado'] = Enfado
    emodict['Triste'] = Triste 
    emodict['Ansiedad'] = Ansiedad
    
    return emodict

  def process(self, dataset):
    # LIWC loading and processing
    parse, category_names = liwc.load_token_parser(self.path)
    result = pd.DataFrame(self.count_emotions(dataset, parse))
    
    return result


# Please use export PYTHONHASHSEED=0 before running this script
# in order to guarantee the reproducibility of the results
seed = 1
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)


# Data loading
path = '../data/HaterNet/'

training_set = pd.read_csv(path + 'train_prep_uncased.tsv', sep='\t')
test_set = pd.read_csv(path + 'test_prep_uncased.tsv', sep='\t')

# Train
x_train = training_set.text
y_train = training_set.label

# Test
x_test = test_set.text
y_test = test_set.label

# Apply Spanish_LIWC to the dataset
spanishLIWC = SpanishLIWC(path='../lexicons/')
liwc_train = spanishLIWC.process(dataset=x_train)
liwc_test = spanishLIWC.process(dataset=x_test)

# Just checking LIWC results
emotions = ['EmoPos', 'EmoNeg', 'Enfado', 'Triste', 'Ansiedad']
for idx in range(1, 4):
  print('Tweet ', idx, ': ', x_train[idx])
  print('Positive emotions: ', liwc_train['EmoPos'][idx])
  print('Negative emotions: ', liwc_train['EmoNeg'][idx])
  print('Anger: ', liwc_train['Enfado'][idx])
  print('Sadness: ', liwc_train['Triste'][idx])
  print('Anxiety: ', liwc_train['Ansiedad'][idx])


# Store each tweet as a list of tokens
token_list = []
for text in x_train:
  token_list.append(preprocessing.text.text_to_word_sequence(text))

# For each list of tokens, store its length
len_texts = []
for index, tweet in enumerate(token_list):
  len_texts.append(len(tweet))

# Calculate statistics
max_value = max(len_texts)  # Use max_seq greater than this value
min_value = min(len_texts)
avg_value = mean(len_texts)
median_value = median(len_texts)

print("El valor max es {} \nEl valor mín es {} \nLa media es {} \nLa mediana es {}".format(max_value, min_value, avg_value, median_value))

max_words = 10000   # Top most frequent words
max_seq = 100       # Size to be padded to (should be greater or equal than the max_value=70)

# Create a tokenize that takes the 10000 most common words
tokenizer = preprocessing.text.Tokenizer(num_words=max_words)

# Fit the tokenizer to the dataset
tokenizer.fit_on_texts(x_train)

# Dictionary ordered by total frequency
word_index = tokenizer.word_index

#print('Found %s unique tokens.' % len(word_index))
print(type(word_index), {k: word_index[k] for k in list(word_index)[:20]})

# Transform each tweet into a numerical sequence
train_sequences = tokenizer.texts_to_sequences(x_train)
test_sequences = tokenizer.texts_to_sequences(x_test)


# Fill each sequence with zeros until max_seq
x_train = preprocessing.sequence.pad_sequences(train_sequences, maxlen=max_seq)
x_test = preprocessing.sequence.pad_sequences(test_sequences, maxlen=max_seq)


# Load FastText embeddings from SUC (300d)
EMBEDDING_DIM = 300
limit = 100000

wordvectors_file_vec = '../embeddings/embeddings-l-model.vec'
w2v_model = KeyedVectors.load_word2vec_format(wordvectors_file_vec, limit=limit, binary=False, encoding='utf8')

def getVector(str):
  if str in w2v_model:
    return w2v_model[str]
  else:
    return None

# Create a matrix with the pretrained embeddings
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = getVector(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# Input layer for the embeddings branch (shape = num_docs, max_seq)
inputA = keras.Input(shape=(max_seq,), dtype='float64', name='Input_A')

# Input layer for the lexicon's branch
inputB = keras.Input(shape=(5,), name='Input_B')

# Embedding layer with Glove's pretrained weights
x = layers.Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            embeddings_initializer=initializers.Constant(embedding_matrix),    # Pretrained weights
                            input_length=max_seq,
                            trainable=False,
                            name='Embedding')(inputA)    # This makes the weights not getting overwritten

# BiLSTM layer with dropout
x = layers.Bidirectional(layers.LSTM(units=128, dropout=0.2),
                                    name='BiLSTM')(x)

# Model for the Embedding branch
model = keras.Model(inputs=inputA, outputs=x, name='Branch_A')

y = layers.Dense(256, activation='relu', name='Dense_Emotions')(inputB)
model = keras.Model(inputs=inputB, outputs=y, name='Branch_B')

# Combines the outputs of both branches
combined = layers.Concatenate()([x, y])

# Forward layer
z = layers.Dense(32, activation='relu', name='Dense_Concatenated')(combined)

# Output binary (sigmoid) classification layer
z = layers.Dense(1, activation='sigmoid', name='Binary_Classifier')(z)

model = keras.Model(inputs=[inputA, inputB], outputs=z, name='Final_Model')

# Model compilation
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# Early stopping and model checkpoint callbacks for fitting
callbacks = [
    EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4),
]

# Model fitting
epochs = 15
history = model.fit([x_train, liwc_train], y_train, epochs=epochs, batch_size=256, validation_split=0.25, verbose=0, callbacks=callbacks)


# Modelo de la última iteracion
y_prob = model.predict([np.array(x_test), liwc_test], batch_size=128, verbose=1)
y_classes = np.around(y_prob, decimals=0)
y_pred = y_classes.astype(int)

print('\nCLASSIFICATION REPORT\n')
print(classification_report(y_test, y_pred))

print('\nCONFUSION MATRIX\n')
print(confusion_matrix(y_test, y_pred))