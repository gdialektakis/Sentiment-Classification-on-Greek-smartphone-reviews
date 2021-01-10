from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression


class MultinomialLogisticRegression:

    def __init__(self, solver, max_iterations):
        self.data = []
        self.model = LogisticRegression(multi_class='multinomial', solver=solver, max_iter=max_iterations)

    def run(self, X_train, X_test, y_train, y_test):
        # train the model
        self.model.fit(X_train, y_train)
        # predict on test data
        y_predicted = self.model.predict(X_test)
        print('Accuracy:  %3f' % accuracy_score(y_predicted, y_test))
        print('Precision:  %3f' % precision_score(y_predicted, y_test, average="macro"))
        print('Recall:  %3f' % recall_score(y_predicted, y_test, average="macro"))
        print('F1_score:  %3f' % f1_score(y_predicted, y_test, average="macro"))
        print("\n")

    def pred(self, test):
        return self.model.predict(test)
