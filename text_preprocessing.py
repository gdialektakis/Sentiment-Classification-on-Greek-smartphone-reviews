import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import model_selection
from sklearn.utils import resample


# spacy.cli.download("el_core_news_md")

bow = CountVectorizer(ngram_range=(1, 2))
tfidf = TfidfVectorizer(ngram_range=(1, 2))


def preprocess():
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
        lemmas = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and len(token.text.strip()) > 0]
        corpus.append(" ".join(lemmas))
    new_df = pd.DataFrame(list(zip(corpus, rating)), columns=['reviews', 'rating'])
    #Remove accents after lemmatization
    replacements = [ ("ά", "α"), ("έ", "ε"), ("ή", "η"), ("ί", "ι"), ("ύ", "υ"), ("ό", "ο"), ("ώ", "ω")]
    for (prev, curr) in replacements:
        new_df['reviews'] = new_df['reviews'].str.replace(prev, curr)
    return new_df


# Need to split dataset in train & test
def bag_of_words(x_train, x_test):
    X_train = bow.fit_transform(x_train)
    x_test = bow.transform(x_test)
    return X_train, x_test


def tf_idf(x_train, x_test):
    # convert th documents into a matrix
    X_train_tfidf = tfidf.fit_transform(x_train)
    X_test_tfidf = tfidf.transform(x_test)
    return X_train_tfidf, X_test_tfidf


def sklearn_train_test(pd_df, sampling=None):
    df = pd_df
    # split documents to train_set and test_set
    if sampling == 'downsample':
        diff_ratings = [5, 4, 3, 2, 1]
        dfs = []
        for dr in diff_ratings:
            curr_df = pd_df[pd_df.rating == dr]
            dfs.append(resample(curr_df, replace=False, n_samples=176, random_state=0))

        df = pd.concat(dfs)

    if sampling == 'oversample':
        diff_ratings = [5, 4, 3, 2, 1]
        dfs = []
        for dr in diff_ratings:
            curr_df = pd_df[pd_df.rating == dr]
            dfs.append(resample(curr_df, replace=True, n_samples=5167, random_state=0))

        df = pd.concat(dfs)

    X = df['reviews']
    y = df['rating']
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=0, stratify=y, test_size=0.1)
    return x_train, x_test, y_train, y_test


def crossValidation(pd_df):
    cv = model_selection.LeaveOneOut()
    X = pd_df['reviews']
    y = pd_df['rating']

    for train_ix, test_ix in cv.split(X):
        x_train, x_test = X[train_ix, :], X[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]
    return x_train, x_test, y_train, y_test