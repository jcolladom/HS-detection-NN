import numpy as np
from gensim.models import KeyedVectors

# Load FastText embeddings from SUC (300d)
def load_suc(path, word_index, emb_dim=300, limit=100000):
    wordvectors_file_vec = path
    w2v_model = KeyedVectors.load_word2vec_format(wordvectors_file_vec, limit=limit, binary=False, encoding='utf8')

    # Create a matrix with the pretrained embeddings
    embedding_matrix = np.zeros((len(word_index) + 1, emb_dim))
    for word, i in word_index.items():
        embedding_vector = _getVector(w2v_model, word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix
    
def _getVector(w2v_model, str):
  if str in w2v_model:
    return w2v_model[str]
  else:
    return None