import text_preprocessing as tp
from multinomialRegression import logistic_regression


if __name__ == "__main__":
    print("Sentiment analysis on Greek Smartphone Reviews")
    pd = tp.preprocess()
    print(pd)

    # split dataset
    x_train, x_test, y_train, y_test = tp.sklearn_train_test(pd)
    # Leave-One-Out Cross-Validation
    # cv_x_train, cv_x_test, cv_y_train, cv_y_test = tp.crossValidation(pd_df)
    X_train_bow, X_test_bow = tp.bag_of_words(x_train, x_test)
    X_train_tf_idf, X_test_tf_idf = tp.tf_idf(x_train, x_test)

    logistic_regression(X_train_bow, X_test_bow, y_train, y_test, 'bag_of_words')
    logistic_regression(X_train_bow, X_test_bow, y_train, y_test, 'tf-idf')