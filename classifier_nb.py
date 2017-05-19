from sklearn.naive_bayes import MultinomialNB

def classify(images, labels):
    mnb = MultinomialNB()
    mnb.fit(images, labels)
    print("Accuracy: ", mnb.score(images, labels))
    return mnb