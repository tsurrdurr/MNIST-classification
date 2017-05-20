from sklearn.neighbors import KNeighborsClassifier

def classify(images, labels):
    nbrs = KNeighborsClassifier(n_neighbors=3)
    nbrs.fit(images, labels)
    print("Accuracy: ", nbrs.score(images, labels))
    return nbrs