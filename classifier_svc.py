from sklearn import svm

def classify(images, labels):
    C = 1.0  # SVM regularization parameter
    lin_svc = svm.LinearSVC(C=C)
    lin_svc.fit(images, labels)
    print("Accuracy: ", lin_svc.score(images, labels))
    return lin_svc