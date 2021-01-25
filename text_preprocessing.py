import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import model_selection
from sklearn.utils import resample
import os.path
import pickle
import numpy as np

from tensorflow.python.keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

# spacy.cli.download("el_core_news_md")

bow = CountVectorizer(ngram_range=(1, 2))
tfidf = TfidfVectorizer(ngram_range=(1, 2))


def preprocess():
    if not os.path.isfile('./preprocessed_data.csv'):
        nlp = spacy.load("el_core_news_md")
        reviews = pd.read_json('./autoscraper/all_reviews.json', encoding='utf8')

        reviews_df = reviews.drop(columns=['phone', 'cons', 'neutral', 'pros'])
        rating = reviews_df['rating']
        replacements = [("κινιτο", "κινητό"), ("\n", " "), (".", " "), ("(", " "), (")", " "), (",", " ")]
        for (prev, curr) in replacements:
            reviews_df['review'] = reviews_df['review'].str.replace(prev, curr)

        reviews_df['review'] = reviews_df['review'].str.replace("\\d", "")
        corpus = []
        for review in reviews_df['review']:
            doc = nlp(review)
            # Tokenize & Remove stop words & punctuation & Lemmatization
            lemmas = [token.lemma_ for token in doc if
                      not token.is_stop and not token.is_punct and len(token.text.strip()) > 0]
            corpus.append(" ".join(lemmas))
        new_df = pd.DataFrame(list(zip(corpus, rating)), columns=['reviews', 'rating'])
        # Remove accents after lemmatization
        replacements = [("ά", "α"), ("έ", "ε"), ("ή", "η"), ("ί", "ι"), ("ύ", "υ"), ("ό", "ο"), ("ώ", "ω")]
        for (prev, curr) in replacements:
            new_df['reviews'] = new_df['reviews'].str.replace(prev, curr)
        new_df.to_csv("./preprocessed_data.csv", index=False)
    else:
        new_df = pd.read_csv("./preprocessed_data.csv")
        print(new_df)
    return new_df


# Need to split dataset in train & test
def bag_of_words(x_train, x_test):
    X_train = bow.fit_transform(x_train)
    X_test = bow.transform(x_test)
    return X_train, X_test


def tf_idf(x_train, x_test):
    # convert th documents into a matrix
    X_train = tfidf.fit_transform(x_train)
    X_test = tfidf.transform(x_test)
    return X_train, X_test


def sklearn_train_test(pd_df, sampling=None):
    df = pd_df
    # split documents to train_set and test_set
    if sampling == 'undersample':
        diff_ratings = [5, 4, 3, 2]
        dfs = []
        max_samples = len(pd_df[pd_df.rating == 1].values)
        for dr in diff_ratings:
            curr_df = pd_df[pd_df.rating == dr]
            dfs.append(resample(curr_df, replace=False, n_samples=max_samples, random_state=0))

        dfs.append(pd_df[pd_df.rating == 1])
        df = pd.concat(dfs)

    X = df['reviews']
    y = df['rating']
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=0, stratify=y, test_size=0.33)

    # Over-sample only on the train data
    new_df = pd.DataFrame(list(zip(x_train, y_train)), columns=['reviews', 'rating'])
    if sampling == 'oversample':
        diff_ratings = [4, 3, 2, 1]
        dfs = []
        max_samples = len(new_df[new_df.rating == 5].values)
        for dr in diff_ratings:
            curr_df = new_df[new_df.rating == dr]
            dfs.append(resample(curr_df, replace=True, n_samples=max_samples, random_state=0))

        dfs.append(new_df[new_df.rating == 5])
        df = pd.concat(dfs)
        x_train = df['reviews']
        y_train = df['rating']

    return x_train, x_test, y_train, y_test


def crossValidation(pd_df):
    cv = model_selection.LeaveOneOut()
    X = pd_df['reviews']
    y = pd_df['rating']

    for train_ix, test_ix in cv.split(X):
        X_train, X_test = X[train_ix, :], X[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]
    return X_train, X_test, y_train, y_test


def lstm_preprocess(df, use_embeddings=False, max_features=2000):
    tokenizer = Tokenizer(num_words=max_features, split=' ')
    tokenizer.fit_on_texts(df['reviews'].values)
    vocab_size = len(tokenizer.word_index) + 1
    X = tokenizer.texts_to_sequences(df['reviews'].values)
    X = sequence.pad_sequences(X)
    embedding_matrix = []
    if use_embeddings:
        # create a weight matrix for words in training docs
        embedding_matrix = np.zeros((vocab_size, 300))
        nlp = spacy.load('el_core_news_md')
        if not os.path.isfile('./embedding_matrix.pickle'):
            for word, i in tokenizer.word_index.items():
                tok = nlp(word)
                if tok.has_vector:
                    embedding_vector = tok.vector
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
            pickle.dump(embedding_matrix, open("embedding_matrix.pickle", "wb"))
        else:
            embedding_matrix = pickle.load(open("embedding_matrix.pickle", "rb"))
    return X, embedding_matrix
