import text_preprocessing as tp

if __name__ == "__main__":
    print("Sentiment analysis on Greek Smartphone Reviews")
    pd = tp.preprocess()
    print(pd)

    bow = tp.bag_of_words(pd)
    print(bow)
