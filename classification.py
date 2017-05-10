from sklearn import svm
import numpy as np
from sklearn.externals import joblib
from read_data import root


images = joblib.load(root + "grayscaled_images_numeric.joblib.pkl")
labels = joblib.load(root + "labels.joblib.pkl")

images = np.asanyarray(images)
images = images.reshape(60000,-60000)
C = 1.0  # SVM regularization parameter
lin_svc = svm.LinearSVC(C=C)
lin_svc.fit(images, labels)

print(lin_svc.score(images, labels))

filename = root + "digits_classifier.joblib.pkl"
joblib.dump(lin_svc, filename)