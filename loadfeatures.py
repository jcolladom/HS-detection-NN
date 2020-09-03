import re
import pandas as pd
import liwc

from collections import Counter

# Spanish LIWC features
class SpanishLIWC:
  def __init__(self, path):
    self.path = path + 'Spanish_LIWC.dic'

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
      tokens = tokenize(tweet)
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
    
    print(emodict)
    
    return emodict

  def process(self, dataset):
    # LIWC loading and processing
    parse, category_names = liwc.load_token_parser(self.path)
    result = pd.DataFrame(self.count_emotions(dataset, parse))
    
    return result
    

# Spanish SEL features
class SEL:
  def __init__(self, path):
    self.path = path + 'SEL.csv'

  # Process the dataset 
  def process(self, dataset):
    sel = pd.read_csv(self.path, ';', header=None)
    words = sel.iloc[:,0].tolist()
    emodict = {'Alegría': [], 'Enojo': [], 'Miedo': [], 'Repulsión': [], 'Sorpresa': [], 'Tristeza': []}
    
    for tweet in dataset:
      sum = {'Alegría': 0.0, 'Enojo': 0.0, 'Miedo': 0.0, 'Repulsión': 0.0, 'Sorpresa': 0.0, 'Tristeza': 0.0}
      tokens = tokenize(tweet)

      for token in tokens:
        matches = [i for i,x in enumerate(words) if x==token]
        for match in matches:
          sum[sel.iloc[match, 2]] += sel.iloc[match, 1]

      for key, value in sum.items():
        emodict[key].append(float(value))

    result = pd.DataFrame(emodict)
    
    return result
  

# Tokenizer function
def tokenize(text):
  # you may want to use a smarter tokenizer
  for match in re.finditer(r'\w+', text, re.UNICODE):
    yield match.group(0)
