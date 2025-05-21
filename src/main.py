# from visualization import plot_sentiment_distribution
# # from data_loader import tokenize, preprocess_data

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import classification_report
# from data_loader import tokenize, preprocess_data
from sklearn.feature_extraction.text import CountVectorizer


def classify_actionability(text):
    tokens = tokenize(text)
    score = 0
    
    good_adjectives = {"helpful", "efficient", "quick", "comfortable", "clean", "affordable", "pleasant", "reliable"}
    bad_adjectives = {"rude", "slow", "uncomfortable", "dirty", "expensive", "faulty", "broken", "unacceptable", "terrible", "horrible"}

    good_adverbs = {"quickly", "efficiently", "politely", "promptly", "smoothly", "easily"}
    bad_adverbs = {"slowly", "rudely", "poorly", "badly", "unfortunately"}

    good_comparatives = {"better", "faster", "more efficient", "easier", "cheaper", "improved"}
    bad_comparatives = {"worse", "slower", "less efficient", "harder", "more expensive", "declined"}

    good_emotional = {"delightful", "pleasant", "great", "fantastic", "amazing", "excellent"}
    bad_emotional = {"frustrating", "annoying", "upsetting", "terrible", "horrible", "disappointing"}

    for t in tokens:
        if t in good_adjectives:
            score +=1
        if t in bad_adjectives:
            score -=1
        if t in good_adverbs:
            score +=1
        if t in bad_adverbs:
            score -=1
        if t in good_comparatives:
            score +=1
        if t in bad_comparatives:
            score -=1
        if t in good_emotional:
            score +=1
        if t in bad_emotional:
            score -=1

    return score


import warnings
warnings.filterwarnings('ignore')

import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
import spacy
import wordcloud
import os # Good for navigating your computer's files
import sys
pd.options.mode.chained_assignment = None #suppress warnings

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from spacy.lang.en.stop_words import STOP_WORDS
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
import locale
locale.getpreferredencoding = lambda: "UTF-8"
import en_core_web_md
text_to_nlp = spacy.load('en_core_web_md')

import scipy
# from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity

def cosine(word1, word2):

  vector1 = word1.reshape(1, -1)
  vector2 = word2.reshape(1, -1)

  return cosine_similarity(vector1, vector2)[0][0]

text_to_nlp = en_core_web_md.load()

def word2vec(word):
    return text_to_nlp(word).vector


import pandas as pd
import re
import nltk

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet


def tokenize_and_embed(text_data):
    """
    Tokenizes the text data and converts it to word embeddings using SpaCy.
    Args:
        text_data (list): A list of text strings to be processed.
    Returns:
        list: A list of lists containing embeddings for each token in each document.
    """
    docs = list(text_to_nlp.pipe(text_data))
    embeddings = [[token.vector for token in doc] for doc in docs]
    return embeddings

def standardize_length(embeddings):
    """
    Ensures all embedding lists are the same length by padding shorter ones with zero vectors.
    Args:
        embeddings (list): A list of lists of embeddings.
    Returns:
        list: A list of lists with padded embeddings to ensure uniform length.
    """
    max_length = max(len(tokens) for tokens in embeddings)
    embedding_dim = len(embeddings[0][0]) if embeddings[0] else 0
    padded_embeddings = [[np.zeros(embedding_dim)] * (max_length - len(tokens)) + tokens for tokens in embeddings]
    return padded_embeddings

def convert_to_array(padded_embeddings):
    """
    Converts a list of padded embeddings into a numpy array.
    Args:
        padded_embeddings (list): A list of lists of padded embeddings.
    Returns:
        numpy.ndarray: A numpy array containing the embeddings suitable for machine learning input.
    """
    return np.array(padded_embeddings)



nltk.download('stopwords')
nltk.download('wordnet')
STOPWORDS = set(stopwords.words('english'))



my_file = open("fictional_dataset.txt", "r")
data = my_file.read()
data_into_list = data.split("\n")

my_file.close

file = open("sentiments.txt", "r")
sent = file.read()
sent_into_list = sent.split("\n")

file.close()

reviews = []
topics = []
# print(data_into_list)
p = 0
for i in data_into_list:
    p+=1
    if (p % 2 == 0):
        reviews.append(i)
    else:
        topics.append(i)


        
        
# print(len(reviews), len(topics), len(sent_into_list))
fictional_dataset_feedback = {"Reviews": reviews, "Topic": topics, "Sentiment": sent_into_list}


df = pd.DataFrame(fictional_dataset_feedback)
df['Actionability Score'] = df["Reviews"].apply(classify_actionability)




from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

class RNNClassifier:
    def __init__(self, num_epochs=30, lstm_units=50, dropout_rate=0.7):
        self.num_epochs = num_epochs
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(LSTM(self.lstm_units, return_sequences=True))
        model.add(Dropout(self.dropout_rate))
        model.add(LSTM(self.lstm_units))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(1, activation='sigmoid'))
        optimizer = Adam(learning_rate=0.001)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):

        if X_train is None and y_train is None:
          print("Arguments are none. Retry with correct arguments.")
          return None
        callbacks = kwargs.pop('callbacks', [])

        if X_val is not None and y_val is not None:
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            callbacks.append(early_stopping)
            return self.model.fit(X_train, y_train, epochs=self.num_epochs, validation_data=(X_val, y_val), callbacks=callbacks, batch_size=32, verbose=1, **kwargs)
        else:
            return self.model.fit(X_train, y_train, epochs=self.num_epochs, batch_size=32, verbose=1, callbacks=callbacks, **kwargs)

    def predict(self, *args, **kwargs):
        predictions = self.model.predict(*args, **kwargs)
        return (predictions > 0.5).astype(int)

    def predict_proba(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def score(self, X, y):
        predictions = self.predict(X)
        return accuracy_score(y, predictions)

    def __getattr__(self, name):
        if name != 'predict' and name != 'predict_proba':
            return getattr(self.model, name)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
#20, 50, 0.5






def generate_model(X_text, y,  epoch, lstm, dropout):
    
    
    X_embeddings = tokenize_and_embed(X_text)  # Tokenize and get embeddings
    
    X_padded = standardize_length(X_embeddings)  # Standardize lengths
    
    X = convert_to_array(X_padded)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1)
    
    
    rnn = RNNClassifier(num_epochs=epoch, lstm_units=lstm, dropout_rate=dropout)
    rnn.fit(X_train, y_train)
    
    y_pred = rnn.predict(X_test)
    accuracy = accuracy_score(y_pred, y_test)
    print(accuracy)

    return rnn

def sent_model():
    z = df['Actionability Score'] 
    
    X_text = df['Reviews']
    y = df['Sentiment']
    y = y.map({"POS": 1, "NEG": 0}).astype('int32')

    return generate_model(X_text, y, 10, 50, 0.3)

def act_model():
    
    z = df['Actionability Score'] 
    
    X_text = df['Reviews']
    
    actionability_model = generate_model(X_text, z, 15, 50, 0.3)
