from sklearn.linear_model import SGDClassifier

def classify(images, labels):
    clf = SGDClassifier(loss="hinge", penalty="l2")
    clf.fit(images, labels)
    print("Accuracy: ", clf.score(images, labels))
    return clf