from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

class Multi_SVM:
    def __init__(self, kernel, gamma, C, degree, decision_function_shape):
        self.data = []
        self.kernel = kernel
        self.gamma = gamma
        self.C = C
        self.degree = degree
        self.decision_function_shape = decision_function_shape
        if kernel == "poly":
            self.model = SVC(kernel=kernel, gamma=gamma, C=C, degree=degree, decision_function_shape=decision_function_shape)
        else:
            self.model = SVC(kernel=kernel, gamma=gamma, C=C, decision_function_shape=decision_function_shape)

    def run(self, x_train, x_test, y_train, y_test):
        self.model.fit(x_train, y_train)
        y_predicted = self.model.predict(x_test)

        accuracy = metrics.accuracy_score(y_test, y_predicted)
        recall = metrics.recall_score(y_test, y_predicted, average="macro")
        precision = metrics.precision_score(y_test, y_predicted, average="macro")
        f1 = metrics.f1_score(y_test, y_predicted, average="macro")

        print("==========================================================")
        print(self.kernel)
        print("SVM Accuracy: %2f" % accuracy)
        print("SVM Precision: %2f" % precision)
        print("SVM Recall: %2f" % recall)
        print("SVM F1 Score: %2f" % f1)

        poly_cm = confusion_matrix(y_test, y_predicted, labels=[1, 2, 3, 4, 5])
        print(poly_cm)
        sns.heatmap(poly_cm)
        plt.xlabel("true label")
        plt.ylabel("predicted label")
        plt.show()
        print("==========================================================")

