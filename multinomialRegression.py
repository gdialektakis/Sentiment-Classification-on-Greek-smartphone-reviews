import text_preprocessing as tp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

data = tp.preprocess()
model = LogisticRegression(multi_class='multinomial')