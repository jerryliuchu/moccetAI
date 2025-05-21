from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from data_loader import tokenize, preprocess_data
from sklearn.feature_extraction.text import CountVectorizer
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import classification_report
# from data_loader import tokenize, preprocess_data
from sklearn.feature_extraction.text import CountVectorizer


def train_sentiment_actionability_model(texts, labels):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(texts)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    return model, vectorizer

def train_topics_model(texts, labels):
    return None



def classify_sentiments(data):
    sentiment_pipeline = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")
    return sentiment_pipeline(data)


    #twenty possible topics: AI Overriding / Ignoring Inputs

# Emotional Frustration / Fatigue with AI

# Positive Collaboration Experience

# AI Creativity and Idea Generation

# AI Repetitive / Redundant Suggestions

# AI Lack of Empathy / Emotional Understanding

# AI Adaptability and Learning

# AI Technical / Analytical Help

# AI Rigidness / Lack of Flexibility

# AI Misunderstanding / Contextual Errors

# AI Performance and Speed Issues

# AI Integration with Workflow / Team

# AI Communication / Tone Mismatch

# AI Assistance with Specific Tasks (e.g., debugging, triage)

# AI Impact on Creativity / Emotional Engagement

# AI Lack of Industry or Domain Knowledge

# AI Causing Negative Outcomes (e.g., client upset)

# AI Support in Meeting Deadlines / Efficiency

# AI Impact on User Confidence / Trust

# General Praise or Satisfaction
    


def topic_num_to_word(lda, feature_names):
    d = {}
    for topic_idx, topic in enumerate(lda.components_):

        top_word_idx = topic.argmax()
        top_word = feature_names[top_word_idx]
        d[topic_idx] = top_word 
    return d
    

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


import pandas as pd
import re
import nltk

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
import openai

openai.my_api_key = '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'

messages = [ {"role": "system", "content": "You are a intelligent assistant."} ]


nltk.download('stopwords')
nltk.download('wordnet')
STOPWORDS = set(stopwords.words('english'))


lemmatizer = WordNetLemmatizer()

def clean_without_tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    t = [lemmatizer.lemmatize(t, wordnet.VERB) for t in tokens]
    return ' '.join(t)

def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    
    #lemmatize 
    return [lemmatizer.lemmatize(t, wordnet.VERB) for t in tokens]







