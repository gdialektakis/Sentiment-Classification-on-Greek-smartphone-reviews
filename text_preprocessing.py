import pandas as pd
import spacy
import time
from sklearn.feature_extraction.text import CountVectorizer

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
def bag_of_words(pd):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(pd['reviews'])
    return X

