from sklearn.utils import class_weight
from tensorflow.python.keras.callbacks import EarlyStopping
from keras import Sequential
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout
from sklearn import model_selection, metrics
from sklearn.metrics import accuracy_score
import numpy as np


class LSTMNetwork:
    def __init__(self, embedding_matrix, use_embeddings=False, is_bidirectional=False, max_features=2000,
                 use_class_weights=False):
        self.use_embeddings = use_embeddings
        self.is_bidirectional = is_bidirectional
        self.embedding_matrix = embedding_matrix
        self.max_features = max_features
        self.use_class_weights = use_class_weights

    def run(self, df, X):
        vocab_size = len(self.embedding_matrix)
        embed_dim = 64
        lstm_out = 32
        model = Sequential()
        if not self.use_embeddings:
            model.add(Embedding(self.max_features, embed_dim, input_length=X.shape[1]))
        else:
            model.add(
                Embedding(vocab_size, 300, weights=[self.embedding_matrix], input_length=X.shape[1], trainable=False))
        if self.is_bidirectional:
            model.add(Bidirectional(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2)))
        else:
            model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(5, activation='softmax'))

        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        Y = df['rating'].values
        Y = np.array([int(v) - 1 for v in Y])

        X_train_data, X_test_data, Y_train_data, Y_test_data = model_selection.train_test_split(X, Y, stratify=Y,
                                                                                                test_size=0.1)
        if self.use_class_weights:
            class_weights = class_weight.compute_class_weight('balanced',
                                                              np.unique(Y_train_data), Y_train_data)
        else:
            class_weights = None

        batch_size = 32
        model.fit(X_train_data,
                  Y_train_data,
                  epochs=4,
                  batch_size=batch_size,
                  validation_split=0.1,
                  shuffle=True,
                  class_weight=class_weights,
                  callbacks=[EarlyStopping(monitor='val_accuracy',
                                           min_delta=0.001,
                                           patience=2,
                                           verbose=1)]
                  )
        predictions_nn_train = model.predict(X_train_data)
        predictions_nn_test = model.predict(X_test_data)

        y_predicted_test = []
        for prediction in predictions_nn_test:
            result = np.argmax(prediction)
            y_predicted_test.append(result)

        y_predicted_train = []

        for prediction in predictions_nn_train:
            result = np.argmax(prediction)
            y_predicted_train.append(result)

        print('Train accuracy:', accuracy_score(Y_train_data, y_predicted_train))
        print('Test accuracy', accuracy_score(Y_test_data, y_predicted_test))
        precision = metrics.precision_score(Y_test_data, y_predicted_test, average="micro")
        recall = metrics.recall_score(Y_test_data, y_predicted_test, average="micro")
        f1 = metrics.f1_score(Y_test_data, y_predicted_test, average="micro")
        print('Test Precision: %2f' % precision)
        print('Test Recall: %2f' % recall)
        print('Test F1: %2f' % f1)
