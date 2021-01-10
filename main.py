import text_preprocessing as tp
import naive_bayes
import multinomialRegression
import SVM

if __name__ == "__main__":
    print("Sentiment analysis on Greek Smartphone Reviews")
    df = tp.preprocess()
    print(df)

    X_train, X_test, y_train, y_test = tp.sklearn_train_test(df, sampling="oversample")
    tfidf_x_train, tfidf_x_test = tp.tf_idf(X_train, X_test)
    bow_x_train, bow_x_test = tp.bag_of_words(X_train, X_test)

    print("Naive Bayes using Bag of Words")
    bow_nb = naive_bayes.NaiveBayes(alpha=0.1)
    bow_nb.run(bow_x_train, bow_x_test, y_train, y_test)
    print("Naive Bayes using TF-IDF")
    tfidf_nb = naive_bayes.NaiveBayes(alpha=0.1)
    tfidf_nb.run(tfidf_x_train, tfidf_x_test, y_train, y_test)

# -------------------------------------------------------------------------------------------------
    """
    print("TF-IDF FOR Polynomial SVM")
    tfidf_poly_svm = SVM.Multi_SVM("poly", 6, 1, 5, "ovo")
    poly_svm_results = tfidf_poly_svm.run(tfidf_x_train, tfidf_x_test, y_train, y_test)
    print("FINISH TF-IDF FOR Polynomial SVM")

    print("TF-IDF FOR Sigmoid SVM")
    tfidf_sigmoid_svm = SVM.Multi_SVM("sigmoid", 2, 10, '', "ovo")
    sigmoid_svm_results = tfidf_sigmoid_svm.run(tfidf_x_train, tfidf_x_test, y_train, y_test)
    print("FINISH TF-IDF FOR Sigmoid SVM \n")

# -------------------------------------------------------------------------------------------------
    """
    print("Multinomial Logistic Regression using Bag of Words")
    bow_lr = multinomialRegression.MultinomialLogisticRegression(solver='saga', max_iterations=200)
    bow_lr.run(bow_x_train, bow_x_test, y_train, y_test)
    bow_lr.feature_importance(bow_x_train, bow_x_test, bow=1)
    print("Multinomial Logistic Regression using TF-IDF")
    tfidf_lr = multinomialRegression.MultinomialLogisticRegression(solver='saga', max_iterations=200)
    tfidf_lr.run(tfidf_x_train, tfidf_x_test, y_train, y_test)
    bow_lr.feature_importance(tfidf_x_train, tfidf_x_test, bow=0)

# -------------------------------------------------------------------------------------------------

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

