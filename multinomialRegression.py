from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression


def logistic_regression(X_train, X_test, y_train, y_test, vectorization):
    model = LogisticRegression(multi_class='multinomial').fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Multinomial Logistic Regression using %s" % vectorization)
    print('Accuracy of Logistic Regression: %s' % accuracy_score(y_pred, y_test))
    print('Precision of Logistic Regression: %s' % precision_score(y_pred, y_test))
    print('Recall of Logistic Regression: %s' % recall_score(y_pred, y_test))
    print('F1_score of Logistic Regression: %s' % f1_score(y_pred, y_test))

    return
