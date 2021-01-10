import text_preprocessing as tp
import naive_bayes
from multinomialRegression import logistic_regression
import SVM

if __name__ == "__main__":
    print("Sentiment analysis on Greek Smartphone Reviews")
    df = tp.preprocess()
    print(df)
    x_train, x_test, y_train, y_test = tp.sklearn_train_test(df, sampling="oversample")

    print("Bag of Words")
    bow_x_train, bow_x_test = tp.bag_of_words(x_train, x_test)
    bow_nb = naive_bayes.NaiveBayes(0.1)
    naive_bayes_results = bow_nb.run(bow_x_train, bow_x_test, y_train, y_test)
    print("TF-IDF")
    tfidf_nb = naive_bayes.NaiveBayes(0.1)
    tfidf_x_train, tfidf_x_test = tp.tf_idf(x_train, x_test)
    naive_bayes_results = tfidf_nb.run(tfidf_x_train, tfidf_x_test, y_train, y_test)

    print("TF-IDF FOR Polynomial SVM")
    tfidf_poly_svm = SVM.Multi_SVM("poly", 6, 1, 5, "ovo")
    tfidf_x_train, tfidf_x_test = tp.tf_idf(x_train, x_test)
    poly_svm_results = tfidf_poly_svm.run(tfidf_x_train, tfidf_x_test, y_train, y_test)
    print("FINISH TF-IDF FOR Polynomial SVM")

    print("TF-IDF FOR Sigmoid SVM")
    tfidf_sigmoid_svm = SVM.Multi_SVM("sigmoid", 2, 10, "ovo")
    tfidf_x_train, tfidf_x_test = tp.tf_idf(x_train, x_test)
    sigmoid_svm_results = tfidf_sigmoid_svm.run(tfidf_x_train, tfidf_x_test, y_train, y_test)
    print("FINISH TF-IDF FOR Sigmoid SVM")

    print("New prediction on the trained model")
    print("====================================")
    test_str = "μετριος κινητο αξιζω αυτο χρηματας υπαρχω παρος πολυ καλυτερα"
    test_bow_transformed = tp.bow.transform([test_str])
    print("B O W")
    print(test_bow_transformed)
    print(bow_nb.pred(test_bow_transformed))
    print("====================================")
    test_tfidf_transformed = tp.tfidf.transform([test_str])
    print("TF-IDF")
    print(test_tfidf_transformed)
    print(tfidf_nb.pred(test_tfidf_transformed))
    print("====================================")

    logistic_regression(bow_x_train, bow_x_test, y_train, y_test, 'bag_of_words')
    logistic_regression(tfidf_x_train, tfidf_x_test, y_train, y_test, 'tf-idf')
