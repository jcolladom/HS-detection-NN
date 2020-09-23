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
                    optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 2e-2, 1e-3, 2e-3])), 
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
                    optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 2e-2, 1e-3, 2e-3])), 
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

        # Pooling
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
                        optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 2e-2, 1e-3, 2e-3])), 
                        metrics=self.metrics)

        return model

# LSTM + features
class LSTMFeaturesModel(HyperModel):
    def __init__(self, vocab_size, max_seq, embedding_matrix, emb_dim, num_emotions, metrics=METRICS):
        self.vocab_size = vocab_size
        self.max_seq = max_seq
        self.embedding_matrix = embedding_matrix
        self.emb_dim = emb_dim
        self.num_emotions = num_emotions
        self.metrics = metrics

    def build(self, hp):
        
        # Input layer for the embeddings branch (shape = num_docs, max_seq)
        inputA = keras.Input(shape=(self.max_seq,), dtype='float64', name='Input_A')

        # Input layer for the lexicon's branch
        '''
        if self.lexicon == 'sel':
            inputB = keras.Input(shape=(6,), name='Input_B')
        elif self.lexicon == 'liwc':
            inputB = keras.Input(shape=(5,), name='Input_B')
        elif self.lexicon == 'emolex':
            inputB = keras.Input(shape=(10,), name='Input_B')
        elif self.lexicon == 'isal':
            inputB = keras.Input(shape=(4,), name='Input_B')
        elif self.lexicon == 'all':
            inputB = keras.Input(shape=(11,), name='Input_B')
        else:
            print("ERROR!!")
            exit()
        '''
        inputB = keras.Input(shape=(self.num_emotions,), name='Input_B')

        # Embedding layer with Glove's pretrained weights
        x = layers.Embedding(self.vocab_size,
                            self.emb_dim,
                            embeddings_initializer=initializers.Constant(self.embedding_matrix),    # Pretrained weights
                            input_length=self.max_seq,
                            trainable=False,
                            name='Embedding')(inputA)    # This makes the weights not getting overwritten

        #x = layers.SpatialDropout1D(rate=hp.Choice('sdo_rate', values=[0.25, 0.5]))(x)

        x = layers.LSTM(units=hp.Int('lstm_units',
                                    min_value=50,
                                    max_value=150,
                                    step=50),
                                return_sequences=False, 
                                dropout=hp.Choice('lstm_do_rate',
                                                values=[0.25, 0.5]))(x)

        # Attention layer
        #x = layers.Attention()(x)

        # Dropout in embeddings' branch
        x = layers.Dropout(rate=hp.Choice('x_do_rate', values=[0.25, 0.5]))(x)

        # Model for the Embedding branch
        model = keras.Model(inputs=inputA, outputs=x, name='Branch_A')

        # Model for the features
        y = layers.Dense(hp.Int('y_dense_units',
                                min_value=32,
                                max_value=512,
                                step=32), 
                            activation=hp.Choice('y_dense_activation', values=['relu', 'tanh']), 
                            kernel_initializer=keras.initializers.glorot_uniform(seed=66))(inputB)
        
        # Dropout in features' branch
        y = layers.Dropout(rate=hp.Choice('y_do_rate', values=[0.25, 0.5]))(y)
        model = keras.Model(inputs=inputB, outputs=y, name='Branch_B')

        # Both branches combination
        combined = layers.Concatenate()([x, y])

        # Forward layer
        z = layers.Dense(hp.Int('dense_units_z',
                                min_value=32,
                                max_value=512,
                                step=32), 
                            activation=hp.Choice('dense_activation_z', values=['relu', 'tanh']), 
                            kernel_initializer=keras.initializers.glorot_uniform(seed=66))(combined)

        # Output binary (sigmoid) classification layer
        z = layers.Dense(1, activation='sigmoid', name='Binary_Classifier')(z)

        model = keras.Model(inputs=[inputA, inputB], outputs=z, name='Final_Model')

        # Model compilation
        model.compile(loss='binary_crossentropy', 
                    optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 2e-2, 1e-3, 2e-3])), 
                    metrics=self.metrics)

        return model