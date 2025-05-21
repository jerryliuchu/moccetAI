import pandas as pd
import re
import nltk

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet



nltk.download('stopwords')
nltk.download('wordnet')
STOPWORDS = set(stopwords.words('english'))




def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    filtered = df[["OverallRating", "ReviewHeader", "ReviewBody"]]
    return filtered

lemmatizer = WordNetLemmatizer()

def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    
    #lemmatize 
    return [lemmatizer.lemmatize(t, wordnet.VERB) for t in tokens]
