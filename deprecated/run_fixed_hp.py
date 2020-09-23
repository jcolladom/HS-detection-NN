# Hyperparameter tuning
# Input parameters (-m model, -t trials)
# Class balancing

import loadembeddings
import buildmodel

import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import re, random, os
import kerastuner

from tensorflow.keras import preprocessing
from statistics import mean, median
from tensorflow import keras
from tensorflow.keras import layers, initializers
from kerastuner.tuners import RandomSearch
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
from gensim.models import KeyedVectors

# Tune hyperparameters
class MyTuner(kerastuner.tuners.RandomSearch):
  def run_trial(self, trial, *args, **kwargs):
    # You can add additional HyperParameters for preprocessing and custom training loops
    # via overriding `run_trial`
    kwargs['batch_size'] = trial.hyperparameters.Choice('batch_size', values=[8, 16, 32, 64, 128, 256])
    #kwargs['epochs'] = trial.hyperparameters.Int('epochs', 100, 500)
    super(MyTuner, self).run_trial(trial, *args, **kwargs)
    
def main(args):
  print("Version", tf.__version__)
  print("Device", tf.test.gpu_device_name())
  print("GPUS", tf.config.list_physical_devices('GPU'))

  # Data loading
  '''path = '../data/HaterNet/'
  training_set = pd.read_csv(path + 'train_prep_uncased.tsv', sep='\t')
  test_set = pd.read_csv(path + 'test_prep_uncased.tsv', sep='\t')
  '''
  path = '../data/HatEval/'
  training_set = pd.read_csv(path + 'train.tsv', sep='\t')
  test_set = pd.read_csv(path + 'test.tsv', sep='\t')
  
  # Train
  '''x_train = training_set.text
  y_train = training_set.label
  '''
  x_train = training_set.iloc[:,3]
  y_train = training_set.iloc[:,1]

  # Test
  '''x_test = test_set.text
  y_test = test_set.label
  '''
  x_test = test_set.iloc[:,3]
  y_test = test_set.iloc[:,1]

  # Store each tweet as a list of tokens
  token_list = []
  for text in x_train:
    token_list.append(preprocessing.text.text_to_word_sequence(text))

  # For each list of tokens, store its length
  len_texts = []
  for index, tweet in enumerate(token_list):
    len_texts.append(len(tweet))

  # Tokenize
  max_words = 10000   # Top most frequent words
  max_seq = 75       # Size to be padded to (should be greater than the max value=70)

  # Create a tokenize that takes the 10000 most common words
  tokenizer = preprocessing.text.Tokenizer(num_words=max_words)

  # Fit the tokenizer to the dataset
  tokenizer.fit_on_texts(x_train)

  # Dictionary ordered by total frequency
  word_index = tokenizer.word_index
  vocab_size = len(word_index) + 1

  # Transform each tweet into a numerical sequence
  train_sequences = tokenizer.texts_to_sequences(x_train)
  test_sequences = tokenizer.texts_to_sequences(x_test)

  # Fill each sequence with zeros until max_seq
  x_train = preprocessing.sequence.pad_sequences(train_sequences, maxlen=max_seq)
  x_test = preprocessing.sequence.pad_sequences(test_sequences, maxlen=max_seq)

  # Load embeddings
  path = '../embeddings/embeddings-l-model.vec'
  EMB_DIM = 300
  LIMIT = 100000
  embedding_matrix = loadembeddings.load_suc(path, word_index, EMB_DIM, LIMIT)

  # Metrics
  METRICS=[
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc')
  ]

  # Create a model instance for the tuner
  if args.model == 'lstm':
    model = buildmodel.LSTMModel(vocab_size, max_seq, embedding_matrix, EMB_DIM, METRICS)
  elif args.model == 'bilstm':
    model = buildmodel.BiLSTMModel(vocab_size, max_seq, embedding_matrix, EMB_DIM, METRICS)
  elif args.model == 'cnn':
    model = buildmodel.CNNModel(vocab_size, max_seq, embedding_matrix, EMB_DIM, METRICS)
  else:
    print("Wrong model. Please choose another one.")
    exit()
  
  # Create the tuner
  tuner = MyTuner(
      model,                                                        # Model's function name
      objective=kerastuner.Objective("accuracy", direction="max"),    # Objective metric
      max_trials=args.trials,                                       # Maximum number of trials
      executions_per_trial=1,                                       # Increase this to reduce results variance
      directory='../hp_trials/',                                    # Directory to store the models
      project_name=args.model,                                      # Project name
      overwrite=True)                                               # Overwrite the project

  class_weights = class_weight.compute_class_weight('balanced',
                                                  np.unique(y_train),
                                                  y_train)

  class_weights = dict(enumerate(class_weights))

  # Early stopping and tensorboard callbacks for fitting
  callbacks = [
      EarlyStopping(monitor='val_loss', mode='max', verbose=0, patience=20)#,
      #keras.callbacks.TensorBoard(log_dir="./logs")
  ]

  epochs = 10
  print("Searching...")
  tuner.search(x_train, y_train, epochs=epochs, validation_split=0.20, verbose=0, callbacks=callbacks)
  # , class_weight=class_weights

  # Save the best model
  best_model = tuner.get_best_models(num_models=1)
  print(tuner.results_summary(num_trials=1))

  # Statistics
  y_prob = best_model[0].predict(np.array(x_test), batch_size=128, verbose=1)
  y_classes = np.around(y_prob, decimals=0)
  y_pred = y_classes.astype(int)

  print('\nCLASSIFICATION REPORT\n')
  print(classification_report(y_test, y_pred, digits=4))

  print('\nCONFUSION MATRIX\n')
  print(confusion_matrix(y_test, y_pred))

  print("Sin CV")
  print("Modelo: " + args.model)
  print("Número de trials: " + str(args.trials))
  print("Número de epochs: " + str(epochs))
  print("Dataset: " + path)

if __name__ == "__main__":
  
  # Use the command below before running this script
  # in order to guarantee reproducibility
  # export PYTHONHASHSEED=0
  
  seed = 1
  np.random.seed(seed)
  random.seed(seed)
  tf.random.set_seed(seed)

  # Args parse
  ap = argparse.ArgumentParser()

  ap.add_argument("-m", 
                  "--model",
                  choices=['lstm','bilstm','cnn'],
                  default='lstm',
                  help="Model to be built")

  ap.add_argument("-t",
                  "--trials",
                  type=int,
                  default=10,
                  help="Number of trials")


  args = ap.parse_args()
  print("Model chosen: " + str(args.model))
  main(args)