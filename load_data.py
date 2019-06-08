import json

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

import pandas as pd
import tensorflow as tf
from tensorflow import keras

from model import implement_model

# breaks down each headline into blocks of lemmatized words, with common stopwords removed
def lemmatize_headline(headline):
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    tokenizer =  RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(headline)
    new_headline = []
    for w in words:
        if w not in stop_words:
            new_headline.append(lemmatizer.lemmatize(w))
    return new_headline

#generates dictionary mapping of words to numbers
def generate_mapping(headlines):
    index = 0
    map = {}
    for h in headlines:
        for word in h:
            if word not in map:
                map[word] = index
                index += 1
    return map

def main():
    PATH = "news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json"
    data = pd.read_json(PATH,lines=True)
    #for now, let's not consider the article link
    data.pop('article_link')
    headlines = data.pop('headline')
    labels = data.pop('is_sarcastic')
    headlines = headlines.apply(lemmatize_headline)

    mapping = generate_mapping(headlines)
    headlines = headlines.apply(lambda h: [mapping[w] for w in h])

    vocab_size = len(mapping)
    model_obj = implement_model(headlines,labels,vocab_size)

if __name__ == "__main__":
    main()
