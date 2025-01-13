from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

def LemmatizeFunc(text):
    lemma=WordNetLemmatizer()
    final=[lemma.lemmatize(word) for word in text]
    return final

def StemmerFunc(text):
    stemmer=PorterStemmer()
    stemmed=[stemmer.stem(word) for word in text]
    return stemmed