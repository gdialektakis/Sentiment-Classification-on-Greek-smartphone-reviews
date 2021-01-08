import pandas as pd
import spacy
import time
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import model_selection

# spacy.cli.download("el_core_news_md")


def preprocess():
    nlp = spacy.load("el_core_news_md")
    reviews = pd.read_json('./autoscraper/all_reviews.json', encoding='utf8')

    reviews_df = reviews.drop(columns=['phone', 'cons', 'neutral', 'pros'])
    rating = reviews_df['rating']
    replacements = [
                      # ("ά", "α"), ("έ", "ε"), ("ή", "η"), ("ί", "ι"), ("ύ", "υ"), ("ό", "ο"), ("ώ", "ω"),
                        ("κινιτο", "κινητό"), ("\n", " "), (".", " "), ("(", " "), (")", " "), (",", " ")
                    ]
    for (prev, curr) in replacements:
        reviews_df['review'] = reviews_df['review'].str.replace(prev, curr)
    corpus = []
    for review in reviews_df['review']:
        doc = nlp(review)
        # Tokenize & Remove stop words & punctuation & Lemmatization
        lemmas = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and len(token.text.strip()) > 0]
        corpus.append(" ".join(lemmas))
    return pd.DataFrame(list(zip(corpus, rating)), columns=['reviews', 'rating'])


# Need to split dataset in train & test
def bag_of_words(x_train, x_test):
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)
    return X_train, x_test

def tf_idf(x_train, x_test):
    # convert th documents into a matrix
    tfidfvectorizer = TfidfVectorizer()
    X_train_tfidf = tfidfvectorizer.fit_transform(x_train)
    X_test_tfidf = tfidfvectorizer.transform(x_test)
    return X_train_tfidf, X_test_tfidf


def sklearn_train_test(pd_df):
    # split documents to train_set and test_set
    X = pd_df['reviews']
    y = pd_df['rating']
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=0)
    return x_train, x_test, y_train, y_test

def crossValidation(pd_df):
    cv = model_selection.LeaveOneOut()
    X = pd_df['reviews']
    y = pd_df['rating']

    for train_ix, test_ix in cv.split(X):
        x_train, x_test = X[train_ix, :], X[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]

    return x_train, x_test, y_train, y_test