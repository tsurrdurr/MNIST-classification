import numpy as np
import classifier_svc as SVC
import classifier_sgd as SGD
import classifier_nb as NB
import classifier_kn as KN
from sklearn.externals import joblib
from read_data import root
import sys

def main():
    if sys.argv[1:]:
        argument = sys.argv[1]
    else:
        argument = None
    if(argument == "-svc"):
        images, labels = read_data()
        svc_classifier = SVC.classify(images, labels)
        dump_classifier(svc_classifier)
    elif(argument == "-sgd"):
        images, labels = read_data()
        sgd_classifier = SGD.classify(images, labels)
        dump_classifier(sgd_classifier)
    elif(argument == "-nb"):
        images, labels = read_data()
        nb_classifier = NB.classify(images, labels)
        dump_classifier(nb_classifier)
    elif(argument == "-kn"):
        images, labels = read_data()
        kn_classifier = KN.classify(images, labels)
        dump_classifier(kn_classifier)
    elif((argument == "-h") | (argument == "-help")):
        print_help()
    else:
        print("Incorrect argument. Try \"-help\"")

def dump_classifier(classifier):
    filename = root + "digits_classifier.joblib.pkl"
    joblib.dump(classifier, filename)


def read_data():
    print("Reading data to form a classifier...")
    images = joblib.load(root + "grayscaled_images_numeric.joblib.pkl")
    labels = joblib.load(root + "labels.joblib.pkl")
    images = np.asanyarray(images)
    images = images.reshape(60000, -60000)
    print("Creating classifier...")
    return images, labels

def print_help():
    print("Possible arguments: \"-svc\", \"-sgd\", \"-nb\"")
    print("\"-svc\" - use LinearSVC")
    print("\"-sgd\" - use SGDClassifier")
    print("\"-nb\" - use MultinomialNB")
    print("\"-nn\" - use MultinomialNB")
if  __name__ =='__main__':main()