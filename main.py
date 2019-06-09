import json
import os
import ast
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

import pandas as pd
import numpy as np

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
    #start at 1 - reserve 0 for padding
    index = 1
    map = {}
    for h in headlines:
        for word in h:
            if word not in map:
                map[word] = index
                index += 1
    return map

def main():
    PATH = "news-headlines-dataset-for-sarcasm-detection/"
    file = "Sarcasm_Headlines_Dataset.json"
    cached_file = "cached_mapped_headlines.csv"

    data = pd.read_json(PATH + file,lines=True)
    #for now, let's not consider the article link
    data.pop('article_link')

    labels = data.pop('is_sarcastic')


    if os.path.isfile(PATH+cached_file):
        headlines = pd.read_csv(PATH+cached_file,header=None,names=['0'],index_col=0,squeeze=True,converters={'0':ast.literal_eval})
        m = 0
        for h in headlines:
            try:
                mx = max(h)
                if mx > m:
                    m = mx
            except:
                pass
        vocab_size=m +1
    else:
        data.dropna()
        headlines = data.pop('headline')
        headlines = headlines.apply(lemmatize_headline)
        print(headlines.tail())
        mapping = generate_mapping(headlines)
        headlines = headlines.apply(lambda h: [mapping[w] for w in h])
        vocab_size = len(mapping) + 1

        headlines.to_csv(PATH + cached_file,header=False)

    model_obj = implement_model(headlines,labels,vocab_size)
    model_obj.plot_hist()
if __name__ == "__main__":
    main()
