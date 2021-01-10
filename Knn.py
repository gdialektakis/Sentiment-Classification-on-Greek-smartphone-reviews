from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
class KNN_Classifier:
    def __init__(self, n_neighbors):
        self.data = []
        self.n_neighbors = n_neighbors
        self.model = KNeighborsClassifier(n_neighbors)

    def run(self, x_train, x_test, y_train, y_test):
        self.model.fit(x_train, y_train)
        y_predicted = self.model.predict(x_test)

        accuracy = metrics.accuracy_score(y_test, y_predicted)
        recall = metrics.recall_score(y_test, y_predicted, average="micro")
        precision = metrics.precision_score(y_test, y_predicted, average="micro")
        f1 = metrics.f1_score(y_test, y_predicted, average="micro")
        jaccard = metrics.jaccard_score(y_test, y_predicted, average="micro")
        classification_report = metrics.classification_report(y_test, y_predicted,
                                                              target_names=['class 1', 'class 2', 'class 3', 'class 4', 'class 5'])
        fpr = {}
        tpr = {}
        thresh = {}
        n_class = 5

        for i in range(n_class):
            fpr[i], tpr[i], thresh[i] = metrics.roc_curve(y_test, y_predicted, pos_label=i)
            print("Class %s" % str(i+1), "ROC AUC: %2f" % metrics.auc(fpr[i], tpr[i]))

        # plotting
        plt.plot(fpr[0], tpr[0], linestyle='--', color='orange', label='Class 1 vs Rest')
        plt.plot(fpr[1], tpr[1], linestyle='--', color='green', label='Class 2 vs Rest')
        plt.plot(fpr[2], tpr[2], linestyle='--', color='blue', label='Class 3 vs Rest')
        plt.plot(fpr[3], tpr[3], linestyle='--', color='red', label='Class 4 vs Rest')
        plt.plot(fpr[4], tpr[4], linestyle='--', color='yellow', label='Class 5 vs Rest')
        plt.title('Multiclass ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive rate')
        plt.legend(loc='best')
        plt.savefig('KNN Multiclass ROC', dpi=300)


        print("==========================================================")
        print("KNN Accuracy: %2f" % accuracy)
        print("KNN Precision: %2f" % precision)
        print("KNN Recall: %2f" % recall)
        print("KNN F1 Score: %2f" % f1)
        print("KNN Jaccard_similarity_score: %2f" % jaccard)
        print("KNN classification_report:")
        print(classification_report)


        cm = confusion_matrix(y_test, y_predicted, labels=[1, 2, 3, 4, 5])
        print(cm)
        sns.heatmap(cm)
        plt.xlabel("true label")
        plt.ylabel("predicted label")
        plt.show()
        print("==========================================================")