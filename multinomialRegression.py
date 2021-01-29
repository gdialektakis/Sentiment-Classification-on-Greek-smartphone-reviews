from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
# import shap
import text_preprocessing as tp

class MultinomialLogisticRegression:

    def __init__(self, solver, max_iterations):
        # shap.initjs()
        self.data = []
        self.model = LogisticRegression(multi_class='multinomial', solver=solver, max_iter=max_iterations)

    def run(self, X_train, X_test, y_train, y_test):
        # train the model
        self.model.fit(X_train, y_train)
        # predict on test data
        y_predicted = self.model.predict(X_test)
        print('Accuracy:  %3f' % accuracy_score(y_predicted, y_test))
        print('Precision:  %3f' % precision_score(y_predicted, y_test, average="weighted"))
        print('Recall:  %3f' % recall_score(y_predicted, y_test, average="weighted"))
        print('F1_score:  %3f' % f1_score(y_predicted, y_test, average="weighted"))
        print("\n")

    # Used for visualization
    def feature_importance(self, X_train, X_test, bow):
        # take first 1000 samples
        # X_test = X_test[1:1000, :]
        # explainer = shap.LinearExplainer(self.model, X_train, feature_perturbation="interventional")
        # shap_values = explainer.shap_values(X_test)
        # X_test_array = X_test.toarray()
        # if bow:
        #     shap.summary_plot(shap_values, X_test_array, feature_names=tp.bow.get_feature_names())
        # else:
        #     shap.summary_plot(shap_values, X_test_array, feature_names=tp.tfidf.get_feature_names())
        pass

    def pred(self, test):
        return self.model.predict(test)
