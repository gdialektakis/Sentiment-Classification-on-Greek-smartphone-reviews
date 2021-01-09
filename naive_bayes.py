from sklearn import datasets, metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class NaiveBayes:
    def __init__(self, alpha):
        self.data = []
        self.alpha = alpha
        self.model = MultinomialNB(alpha=alpha)

    def run(self, x_train, x_test, y_train, y_test):
        self.model.fit(x_train, y_train)
        y_predicted = self.model.predict(x_test)

        accuracy = metrics.accuracy_score(y_test, y_predicted)
        precision = metrics.precision_score(y_test, y_predicted, average="macro")
        recall = metrics.recall_score(y_test, y_predicted, average="macro")
        f1 = metrics.f1_score(y_test, y_predicted, average="macro")
        print('Accuracy: %2f' % accuracy)
        print('Precision: %2f' % precision)
        print('Recall: %2f' % recall)
        print('F1: %2f' % f1)
        cm = confusion_matrix(y_test, y_predicted, labels=[1, 2, 3, 4, 5])
        print(cm)
        sns.heatmap(cm)
        plt.xlabel('true label')
        plt.ylabel('predicted label')
        plt.show()

    def pred(self, test):
        return self.model.predict(test)
