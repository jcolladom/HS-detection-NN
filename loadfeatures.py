import re
import pandas as pd
import liwc
import unicodedata

from nltk.tokenize import TweetTokenizer
from collections import Counter

# Spanish LIWC lexicon
class SpanishLIWC:
  def __init__(self, path):
    self.path = path + 'Spanish_LIWC_Sin_Tildes.dic'

  # Emotions counter for the LIWC lexicon
  # Returns: a dictionary with emotions as keys and its frequency on each tweet as value
  def count_emotions(self, dataset, parse):
    EmoPos = []
    EmoNeg = []
    Enfado = []
    Triste = []
    Ansiedad = []
    emodict = {}

    words_matched = 0

    for tweet in dataset:
      tokens = tokenize(tweet)
      num_tokens = len(list(tokens))
      
      tokens_count = Counter(category for token in tokens for category in parse(token))

      words_matched += tokens_count['EmoPos']
      words_matched += tokens_count['EmoNeg']
      words_matched += tokens_count['Enfado']
      words_matched += tokens_count['Triste']
      words_matched += tokens_count['Ansiedad']

      EmoPos.append(tokens_count['EmoPos'] / num_tokens)
      EmoNeg.append(tokens_count['EmoNeg'] / num_tokens)
      Enfado.append(tokens_count['Enfado'] / num_tokens)
      Triste.append(tokens_count['Triste'] / num_tokens)
      Ansiedad.append(tokens_count['Ansiedad'] / num_tokens)

    print("Words matched: " + str(words_matched))
    emodict['EmoPos'] = EmoPos
    emodict['EmoNeg'] = EmoNeg
    emodict['Enfado'] = Enfado
    emodict['Tristeza'] = Triste 
    emodict['Ansiedad'] = Ansiedad
    
    return emodict

  def process(self, dataset):
    print("Processing LIWC...")

    # LIWC loading and processing
    parse, category_names = liwc.load_token_parser(self.path)
    result = pd.DataFrame(self.count_emotions(dataset, parse))
    return result
    

# Spanish SEL lexicon
class SEL:
  def __init__(self, path):
    self.path = path + 'SEL.csv'

  # Process the dataset 
  def process(self, dataset):
    print("Processing SEL...")
    sel = pd.read_csv(self.path, ';', header=None)
    words = sel.iloc[:,0].tolist()
    emodict = {'Alegría': [], 'Enojo': [], 'Miedo': [], 'Repulsión': [], 'Sorpresa': [], 'Tristeza': []}
    
    words_matched = []

    for tweet in dataset:
      sum = {'Alegría': 0.0, 'Enojo': 0.0, 'Miedo': 0.0, 'Repulsión': 0.0, 'Sorpresa': 0.0, 'Tristeza': 0.0}
      tokens = tokenize(tweet)
      num_tokens = len(list(tokens))
      
      for token in tokens:
        matches = [i for i,x in enumerate(words) if x==token]
        for match in matches:
          words_matched.append(sel.iloc[match, 0])
          sum[sel.iloc[match, 2]] += sel.iloc[match, 1]
      
      for key, value in sum.items():
        emodict[key].append(float(value) / num_tokens)

    #print(words_matched)
    print("Words matched: " + str(len(words_matched)))

    # Standarize keys in order to match them with other lexicons
    emodict['Alegria'] = emodict.pop('Alegría')
    emodict['Enfado'] = emodict.pop('Enojo')
    emodict['Repulsion'] = emodict.pop('Repulsión')

    result = pd.DataFrame(emodict)
    return result
  

# Spanish iSAL lexicon
class iSAL:
  def __init__(self, path):
    self.path = path + 'iSALv2m.csv'

  # Process the dataset 
  def process(self, dataset):
    print("Processing iSAL...")
    lex = pd.read_csv(self.path, sep='\t', header=0, decimal=',')
    emodict = {'anger': [], 'fear': [], 'sadness': [], 'joy': []}
    
    # Remove accents from the lexicon
    acc_words = lex.iloc[:,0].tolist()
    words = []
    for word in acc_words:
      words.append(unicodedata.normalize('NFKD', word).encode('ASCII', 'ignore').decode("utf-8"))
    
    words_matched = []

    for tweet in dataset:
      sum = {'anger': 0.0, 'fear': 0.0, 'sadness': 0.0, 'joy': 0.0}
      tokens = tokenize(tweet)
      num_tokens = len(list(tokens))
      
      for token in tokens:
        matches = [i for i,x in enumerate(words) if x==token]
        for match in matches:
          words_matched.append(lex.iloc[match, 0])
          sum[lex.iloc[match, 2]] += lex.iloc[match, 1]
      
      for key, value in sum.items():
        emodict[key].append(float(value) / num_tokens)

    print("Words matched: " + str(len(words_matched)))

    # Standarize keys in order to match them with other lexicons
    emodict['Enfado'] = emodict.pop('anger')
    emodict['Miedo'] = emodict.pop('fear')
    emodict['Tristeza'] = emodict.pop('sadness')
    emodict['Alegria'] = emodict.pop('joy')

    result = pd.DataFrame(emodict)
    return result
  

# Emolex lexicon
class Emolex:
  def __init__(self, path):
    self.path = path + 'Emolex.xlsx'

  # Process the dataset 
  def process(self, dataset):
    print("Processing Emolex...")
    # Load Spanish column (CI) and every emotion (DB:DK)
    lex = pd.read_excel(self.path, usecols="CI,DB:DK")
    
    # Drop invalid values (NO TRANSLATION)
    lex = lex[lex["Spanish (es)"] != "NO TRANSLATION"]

    # Remove accents from the lexicon
    acc_words = lex.iloc[:,0].tolist()
    words = []
    for word in acc_words:
      words.append(unicodedata.normalize('NFKD', word).encode('ASCII', 'ignore').decode("utf-8"))

    # Stores each emotion (10)
    emodict = {'Positive': [], 'Negative': [], 'Anger': [], 'Anticipation': [], 'Disgust': [], 'Fear': [], 'Joy': [], 'Sadness': [], 'Surprise': [], 'Trust': []}
    
    words_matched = []

    for tweet in dataset:
      sum = {'Positive': 0.0, 'Negative': 0.0, 'Anger': 0.0, 'Anticipation': 0.0, 'Disgust': 0.0, 'Fear': 0.0, 'Joy': 0.0, 'Sadness': 0.0, 'Surprise': 0.0, 'Trust': 0.0}
      tokens = tokenize(tweet)
      num_tokens = len(list(tokens))
      
      for token in tokens:
        matches = [i for i,x in enumerate(words) if x==token]
        for match in matches:
          words_matched.append(lex.iloc[match, 0])
          for key in sum.keys():
            sum[key] = lex.iloc[match][key]
      
      for key, value in sum.items():
        emodict[key].append(float(value) / num_tokens)

    #print(words_matched)
    print("Words matched: " + str(len(words_matched)))

    # Standarize keys in order to match them with other lexicons
    emodict['EmoPos'] = emodict.pop('Positive')
    emodict['EmoNeg'] = emodict.pop('Negative')
    emodict['Enfado'] = emodict.pop('Anger')
    emodict['Expectacion'] = emodict.pop('Anticipation')
    emodict['Repulsion'] = emodict.pop('Disgust')
    emodict['Miedo'] = emodict.pop('Fear')
    emodict['Alegria'] = emodict.pop('Joy')
    emodict['Tristeza'] = emodict.pop('Sadness')
    emodict['Sorpresa'] = emodict.pop('Surprise')
    emodict['Confianza'] = emodict.pop('Trust')

    result = pd.DataFrame(emodict)
    return result


# Pack every lexicon together
class All:
  def __init__(self, path):
    self.path = path

  def process(self, dataset):
    liwc = SpanishLIWC(path=self.path)
    lex1 = liwc.process(dataset=dataset)

    sel = SEL(path=self.path)
    lex2 = sel.process(dataset=dataset)

    emolex = Emolex(path=self.path)
    lex3 = emolex.process(dataset=dataset)

    isal = iSAL(path=self.path)
    lex4 = isal.process(dataset=dataset)

    emodict = {'EmoPos': [], 'EmoNeg': [], 'Enfado': [], 'Tristeza': [], 'Ansiedad': [], 'Alegria': [], 'Miedo': [], 'Repulsion': [], 'Sorpresa': [], 'Expectacion': [], 'Confianza': []}
    
    # Sums every emotion and normalizes it (divide by the total number of lexicons used, 3 in this case)
    for key in emodict.keys():
      emodict[key] = (lex1.get(key, 0) + lex2.get(key, 0) + lex3.get(key, 0) + lex4.get(key, 0)) / 4
    
    result = pd.DataFrame(emodict)
    return result


# Tokenizer function
def tokenize(text):
  tknzr = TweetTokenizer(preserve_case=False)
  tokens = tknzr.tokenize(text)
  return tokens
  
  #return [match.group(0) for match in re.finditer(r'\w+', text, re.UNICODE)]
  # you may want to use a smarter tokenizer
  #for match in re.finditer(r'\w+', text, re.UNICODE):
  #  yield match.group(0)
