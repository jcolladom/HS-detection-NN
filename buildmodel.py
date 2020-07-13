from tensorflow import keras
from tensorflow.keras import layers, initializers
from kerastuner import HyperModel

# Metrics
METRICS=[
	keras.metrics.BinaryAccuracy(name='accuracy'),
	keras.metrics.Precision(name='precision'),
	keras.metrics.Recall(name='recall'),
	keras.metrics.AUC(name='auc')
]

class LSTMModel(HyperModel):

    def __init__(self, vocab_size, max_seq, embedding_matrix, emb_dim, metrics=METRICS):
        self.vocab_size = vocab_size
        self.max_seq = max_seq
        self.embedding_matrix = embedding_matrix
        self.emb_dim = emb_dim
        self.metrics = metrics

    def build(self, hp):
        
        # Input layer (shape = num_docs, max_seq)
        inputs = keras.Input(shape=(self.max_seq,), dtype='float64', name='Input')

        # Embedding layer with pretrained weights
        embedding = layers.Embedding(self.vocab_size,
                                    self.emb_dim,
                                    embeddings_initializer=initializers.Constant(self.embedding_matrix),    # Pretrained weights
                                    input_length=self.max_seq,
                                    trainable=False,
                                    name='Embedding')(inputs)    # This makes the weights not getting overwritten

        # Dropout
        embedding = keras.layers.SpatialDropout1D(rate=hp.Choice('sdo_rate', values=[0.25, 0.5]))(embedding)

        # BiLSTM layer with dropout
        activations = layers.LSTM(units=hp.Int('lstm_units',
                                                min_value=50,
                                                max_value=150,
                                                step=50),
                                return_sequences=False, 
                                dropout=hp.Choice('lstm_do_rate',
                                                values=[0.25, 0.5]))(embedding)
        # Dense layer
        dense = layers.Dense(hp.Int('dense_units',
                                    min_value=32,
                                    max_value=512,
                                    step=32), 
                            activation=hp.Choice('dense_activation', values=['relu', 'tanh']), 
                            kernel_initializer=keras.initializers.glorot_uniform(seed=66))(activations)
    
        # Dropout layer
        dropout = layers.Dropout(rate=hp.Choice('do_rate', values=[0.25, 0.5]))(dense)

        # Output binary (sigmoid) classification layer
        x = layers.Dense(1, activation='sigmoid', name='Binary_Classifier')(dropout)

        # Model compilation
        model = keras.Model(inputs=inputs, outputs=x, name='functional_model')
        model.compile(loss='binary_crossentropy', 
                    optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 2e-3, 1e-4])), 
                    metrics=self.metrics)

        return model

# BiLSTM model
class BiLSTMModel(HyperModel):

    def __init__(self, vocab_size, max_seq, embedding_matrix, emb_dim, metrics=METRICS):
        self.vocab_size = vocab_size
        self.max_seq = max_seq
        self.embedding_matrix = embedding_matrix
        self.emb_dim = emb_dim
        self.metrics = metrics

    
    def build(self, hp):
        # Input layer (shape = num_docs, max_seq)
        inputs = keras.Input(shape=(self.max_seq,), dtype='float64', name='Input')

        # Embedding layer with pretrained weights
        embedding = layers.Embedding(self.vocab_size,
                                    self.emb_dim,
                                    embeddings_initializer=initializers.Constant(self.embedding_matrix),    # Pretrained weights
                                    input_length=self.max_seq,
                                    trainable=False,
                                    name='Embedding')(inputs)    # This makes the weights not getting overwritten

        # Dropout
        embedding = keras.layers.SpatialDropout1D(rate=hp.Choice('sdo_rate', values=[0.25, 0.5]))(embedding)

        # BiLSTM layer with dropout
        activations = layers.Bidirectional(layers.LSTM(units=hp.Int('lstm_units',
                                                                    min_value=50,
                                                                    max_value=150,
                                                                    step=50),
                                                        return_sequences=False, 
                                                        dropout=hp.Choice('lstm_do_rate',
                                                                        values=[0.25, 0.5])))(embedding)
        # Dense layer
        dense = layers.Dense(hp.Int('dense_units',
                                    min_value=32,
                                    max_value=512,
                                    step=32), 
                            activation=hp.Choice('dense_activation', values=['relu', 'tanh']), 
                            kernel_initializer=keras.initializers.glorot_uniform(seed=66))(activations)
    
        # Dropout layer
        dropout = layers.Dropout(rate=hp.Choice('do_rate', values=[0.25, 0.5]))(dense)

        # Output binary (sigmoid) classification layer
        x = layers.Dense(1, activation='sigmoid', name='Binary_Classifier')(dropout)

        # Model compilation
        model = keras.Model(inputs=inputs, outputs=x, name='functional_model')
        model.compile(loss='binary_crossentropy', 
                    optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 2e-3, 1e-4])), 
                    metrics=self.metrics)

        return model

# CNN model
class CNNModel(HyperModel):

    def __init__(self, vocab_size, max_seq, embedding_matrix, emb_dim, metrics=METRICS):
        self.vocab_size = vocab_size
        self.max_seq = max_seq
        self.embedding_matrix = embedding_matrix
        self.emb_dim = emb_dim
        self.metrics = metrics

    def build(self, hp):
        # Input layer (shape = num_docs, max_seq)
        inputs = keras.Input(shape=(self.max_seq,), dtype='float64', name='Input')

        # Embedding layer with pretrained weights
        embedding = layers.Embedding(self.vocab_size,
                                    self.emb_dim,
                                    embeddings_initializer=initializers.Constant(self.embedding_matrix),    # Pretrained weights
                                    input_length=self.max_seq,
                                    trainable=False,
                                    name='Embedding')(inputs)    # This makes the weights not getting overwritten

        # Dropout
        embedding = keras.layers.SpatialDropout1D(rate=hp.Choice('sdo_rate', values=[0.25, 0.5]))(embedding)
        conv_layer = keras.layers.Convolution1D(hp.Int('conv_size',
                                                        min_value=50,
                                                        max_value=150,
                                                        step=50),
                                                3,
                                                activation=hp.Choice('conv_activation', values=['relu', 'tanh']))(embedding)

        pooling_layer = keras.layers.GlobalMaxPool1D()(conv_layer)


        # Dense layer
        dense = layers.Dense(hp.Int('dense_units',
                                    min_value=32,
                                    max_value=512,
                                    step=32), 
                            activation=hp.Choice('dense_activation', values=['relu', 'tanh']))(pooling_layer)
        
        # Dropout layer
        dropout = layers.Dropout(rate=hp.Choice('do_rate', values=[0.25, 0.5]))(dense)

        # Output binary (sigmoid) classification layer
        x = layers.Dense(1, activation='sigmoid', name='Binary_Classifier')(dropout)

        # Model compilation
        model = keras.Model(inputs=inputs, outputs=x, name='functional_model')
        model.compile(loss='binary_crossentropy', 
                        optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 2e-3, 1e-4])), 
                        metrics=self.metrics)

        return model

# BiLSTM + CNN model
class BiLSTM_CNNModel(HyperModel):

    def __init__(self, vocab_size, max_seq, embedding_matrix, emb_dim, metrics=METRICS):
        self.vocab_size = vocab_size
        self.max_seq = max_seq
        self.embedding_matrix = embedding_matrix
        self.emb_dim = emb_dim
        self.metrics = metrics

    def build(self, hp):
        # Input layer (shape = num_docs, max_seq)
        input_layer = keras.Input(shape=(self.max_seq,), dtype='float64', name='Input')

        # Embedding layer with pretrained weights
        embedding_layer = layers.Embedding(self.vocab_size,
                                    self.emb_dim,
                                    embeddings_initializer=initializers.Constant(self.embedding_matrix),    # Pretrained weights
                                    input_length=self.max_seq,
                                    trainable=False,
                                    name='Embedding')(input_layer)    # This makes the weights not getting overwritten

        # Dropout
        embedding_layer = keras.layers.SpatialDropout1D(rate=hp.Choice('sdo_rate', values=[0.25, 0.5]))(embedding_layer)

        # BiLSTM layer with dropout
        rnn_layer = layers.Bidirectional(layers.LSTM(units=hp.Int('lstm_units',
                                                                    min_value=50,
                                                                    max_value=150,
                                                                    step=50),
                                                        return_sequences=True))(embedding_layer)

        # Convolutional layer
        conv_layer = layers.Convolution1D(hp.Int('conv_size',
                                                    min_value=50,
                                                    max_value=150,
                                                    step=50),
                                            3,
                                            activation=hp.Choice('conv_activation', values=['relu', 'tanh']))(rnn_layer)

        # Pooling layer
        pooling_layer = layers.GlobalMaxPool1D()(conv_layer)

        # Dense layer
        output_layer1 = layers.Dense(hp.Int('dense_units',
                                    min_value=32,
                                    max_value=512,
                                    step=32), 
                            activation=hp.Choice('dense_activation', values=['relu', 'tanh']))(pooling_layer)
        
        # Dropout layer
        output_layer1 = layers.Dropout(rate=hp.Choice('do_rate', values=[0.25, 0.5]))(output_layer1)

        # Output binary (sigmoid) classification layer
        output_layer2 = layers.Dense(1, activation='sigmoid')(output_layer1)

        # Model compilation
        model = keras.Model(inputs=input_layer, outputs=output_layer2, name='functional_model')
        model.compile(loss='binary_crossentropy', 
                        optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 2e-3, 1e-4])), 
                        metrics=self.metrics)

        return model