import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import string
import nltk
import emoji
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


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
    lemma=WordNetLemmatizer()
    work=word_tokenize(text)
    work=[word for word in work if word.lower() not in stop_words]
    final_text=[lemma.lemmatize(word) for word in work]
    return final_text



