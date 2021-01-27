import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense, Embedding, LSTM, Bidirectional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import compute_class_weight


class LSTMNetwork:
    def __init__(self, embedding_matrix, use_embeddings=False, is_bidirectional=False, max_features=2000,
                 use_class_weights=False, type='LSTM'):
        self.use_embeddings = use_embeddings
        self.is_bidirectional = is_bidirectional
        self.embedding_matrix = embedding_matrix
        self.max_features = max_features
        self.use_class_weights = use_class_weights
        self.type = type

    def run(self, df, X):
        vocab_size = len(self.embedding_matrix)
        embed_dim = 64
        lstm_out = 32
        model = Sequential()

        if not self.use_embeddings:
            model.add(Embedding(self.max_features, embed_dim, input_length=X.shape[1]))
        else:
            model.add(
                Embedding(vocab_size, 300, weights=[self.embedding_matrix], input_length=X.shape[1],
                          trainable=True, mask_zero=True))
        if self.is_bidirectional:
            model.add(Bidirectional(LSTM(lstm_out, dropout=0.2)))
        else:
            model.add(LSTM(lstm_out, dropout=0.2))
        model.add(Dense(5, activation='softmax'))

        optimizer = keras.optimizers.Adam(learning_rate=0.01)

        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        print(model.summary())

        Y = df['rating'].values
        Y = np.array([int(v) - 1 for v in Y])
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, test_size=0.1)
        if self.use_class_weights:
            class_weights = compute_class_weight('balanced', np.unique(Y_train), Y_train)
        else:
            class_weights = None

        filepath = "weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        # checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

        history = model.fit(X_train,
                            Y_train,
                            epochs=5,
                            batch_size=64,
                            validation_split=0.1,
                            shuffle=True,
                            class_weight=class_weights,
                            callbacks=[checkpoint]
                            )

        # Plot curve of Cross Entropy Loss
        plt.plot(history.history['loss'], label='training data')
        plt.plot(history.history['val_loss'], label='validation data')
        plt.xticks(np.arange(1, 6, step=1))
        plt.title('%s Cross Entropy Loss' % self.type)
        plt.ylabel('Loss value')
        plt.xlabel('No. epoch')
        plt.legend(loc="upper right")
        plt.show()

        # Plot curve of accuracy
        plt.plot(history.history['accuracy'], label='training data')
        plt.plot(history.history['val_accuracy'], label='validation data')
        plt.xticks(np.arange(1, 6, step=1))
        plt.title('%s Accuracy' % self.type)
        plt.ylabel('Accuracy')
        plt.xlabel('No. epoch')
        plt.legend(loc="lower right")
        plt.show()

        print("Created model and loaded weights from file")
        # load model with best validation accuracy
        model.load_weights("weights.best.hdf5")
        # Compile model (required to make predictions)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # predict using loaded model
        predictions_nn_test = model.predict(X_test)
        y_predicted_test = []
        for prediction in predictions_nn_test:
            result = np.argmax(prediction)
            y_predicted_test.append(result)

        predictions_nn_train = model.predict(X_train)
        y_predicted_train = []
        for prediction in predictions_nn_train:
            result = np.argmax(prediction)
            y_predicted_train.append(result)

        print("\n------------- %s Evaluation -------------" % self.type)
        print('Train accuracy: %.4f' % accuracy_score(Y_train, y_predicted_train))
        print('--------------Evaluation on test set--------------')
        print('Accuracy: %.4f' % accuracy_score(Y_test, y_predicted_test))
        print('Precision: %.4f' % precision_score(Y_test, y_predicted_test, average="weighted"))
        print('Recall: %.4f' % recall_score(Y_test, y_predicted_test, average="weighted"))
        print('F1: %.4f' % f1_score(Y_test, y_predicted_test, average="weighted"))
