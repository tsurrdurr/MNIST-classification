from sklearn import svm
import numpy as np
from sklearn.externals import joblib

images = joblib.load("C:\\temp\\grayscaled_images_numeric")
labels = joblib.load("C:\\temp\\labels")

images = np.asanyarray(images)
images = images.reshape(60000,-60000)
C = 1.0  # SVM regularization parameter
lin_svc = svm.LinearSVC(C=C)
lin_svc.fit(images, labels)

print(lin_svc.score(images, labels))

filename = "C:\\temp\\digits_classifier.joblib.pkl"
_ = joblib.dump(lin_svc, filename)