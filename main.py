import text_preprocessing as tp

if __name__ == "__main__":
    print("Sentiment analysis on Greek Smartphone Reviews")
    pd = tp.preprocess()
    print(pd)

    # split dataset
    x_train, x_test, y_train, y_test = tp.sklearn_train_test(pd_df)
    # Leave-One-Out Cross-Validation
    # cv_x_train, cv_x_test, cv_y_train, cv_y_test = tp.crossValidation(pd_df)
    X_train_bow, x_test_bow = tp.bag_of_words(x_train, x_test)
    X_train_tf_idf, X_test_tf_idf = tp.tf_idf(x_train, x_test)
