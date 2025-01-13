import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import string
import nltk
import emoji
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download("vader_lexicon")
from textblob import TextBlob



def CleanText(text):
    if not isinstance(text,str):
        return text
    text=text.lower()
    text=re.sub(r'\d+', '', text)
    text=text.translate(str.maketrans('', '', string.punctuation))
    text=re.sub(r'\W+', ' ', text)
    return text



def StopWordFunc(text):
    stop_words=set(stopwords.words('english'))
    work=word_tokenize(text)
    work=[word for word in work if word.lower() not in stop_words]
    return work

def list_features(text):
    words=set(text)
    features={}
    for word in words:
        features['contains({})'.format(word)]=(word in words)
    return features

def sentimentanalyzervader(text):
  analyzer=SentimentIntensityAnalyzer()
  return analyzer.polarity_scores(text)

def sentimentTextBlob(text):
  return TextBlob(text).sentiment.subjectivity

def sentimentscoreTB(score):
  if score > 0:
    return 'positive'
  elif score < 0:
    return 'negative'
  else:
    return 'neutral'