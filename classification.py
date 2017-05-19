import numpy as np
import classifier_svc as SVC
import classifier_sgd as SGD
from sklearn.externals import joblib
from read_data import root
import sys

def main():
    argument = sys.argv[1]
    if(argument == "-svc"):
        images, labels = read_data()
        svc_classifier = SVC.classify(images, labels)
        dump_classifier(svc_classifier)
    elif(argument == "-sgd"):
        images, labels = read_data()
        sgd_classifier = SGD.classify(images, labels)
        dump_classifier(sgd_classifier)
    else:
        print("Incorrect argument. Possible arguments: \"-svc\", \"-sgd\"")

def dump_classifier(classifier):
    filename = root + "digits_classifier.joblib.pkl"
    joblib.dump(classifier, filename)


def read_data():
    print("Reading data to form a classifier...")
    images = joblib.load(root + "grayscaled_images_numeric.joblib.pkl")
    labels = joblib.load(root + "labels.joblib.pkl")
    images = np.asanyarray(images)
    images = images.reshape(60000, -60000)
    print("Creating classificator...")
    return images, labels


if  __name__ =='__main__':main()