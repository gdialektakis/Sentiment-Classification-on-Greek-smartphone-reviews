import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer

# spacy.cli.download("el_core_news_md")

nlp = spacy.load("el_core_news_md")
reviews = pd.read_json('./autoscraper/all_reviews.json', encoding='utf8')

reviews = reviews.drop(columns=['phone', 'cons', 'neutral', 'pros'])
x = reviews['review']
print("==============================================")
# Bag Of Words
review = x
replacements = [("ά", "α"), ("έ", "ε"), ("ή", "η"), ("ί", "ι"), ("ύ", "υ"), ("ό", "ο"), ("ώ", "ω"), ("κινιτο", "κινητο"), ("\n", "")]
for acc in replacements:
    prev, curr = acc
    review = review.str.replace(prev, curr)

docs = []
corpus = []

# for r in review:
#     doc = nlp(r)
#     # Tokenize & Remove stop words & punctuation
#     filtered_tokens = [token for token in doc if not token.is_stop and not token.is_punct]
#
#     # Lemmatization
#     lemmas = [token.lemma_ for token in filtered_tokens]
#     # corpus.extend(lemmas)

# vec = CountVectorizer().fit(corpus)
# X = vec.transform(corpus)
# sum_words = X.sum(axis=0)
# words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
# words_freq = sorted(words_freq, key=lambda y: y[1], reverse=True)
# print(words_freq)
doc = nlp(review[1])
# Tokenize
token_list = [token for token in doc]
# Remove stop words & punctuation
filtered_tokens = [token for token in doc if not token.is_stop and not token.is_punct]

# Lemmatization
lemmas = [
    f"Token: {token}, lemma: {token.lemma_}"
    for token in filtered_tokens
]

print(doc)
print(token_list)
print(filtered_tokens)
print(lemmas)

print("================================")
print("Example Word Vector")
# Get the word
print()
doc = nlp("99ε")
token_list = [token for token in doc]
print(token_list[0].vector)

print(len(nlp.vocab.vectors.keys()))
print("================================")
print("================================")

# for token in doc:
#     print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
#             token.shape_, token.is_alpha, token.is_stop)
