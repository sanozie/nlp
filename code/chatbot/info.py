from nltk.corpus import wordnet as wn
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split

stoplist = set(stopwords.words('english'))


def generate_df():
    df = pd.read_csv('Stereotypes.csv', usecols=[2, 14, 15, 16, 17, 18])
    df['tv'].replace('', np.nan, inplace=True)
    df.dropna(inplace=True)
    df['tv'].str.lower()
    return df


def net_of_words(word, table):
    synsets = wn.synsets(word)
    # print(synsets)
    for synset in synsets:
        hyp = synset.hypernyms()[0]
        top = wn.synset('entity.n.01')

        while hyp:
            test = hyp.lemma_names()[0]
            if test in table:
                return test
            if hyp == top:
                break
            if hyp.hypernyms():
                hyp = hyp.hypernyms()[0]


def train(df, col):
    vectorizer = TfidfVectorizer(stop_words=stoplist, binary=True)
    X = vectorizer.fit_transform(df.tv)
    df[col] = df[col].astype('int')
    y = df[col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1, train_size=7017, random_state=1234)

    classifier = BernoulliNB()
    classifier.fit(X_train, y_train)
    pred = classifier.predict(X_test)

    return classifier, vectorizer

