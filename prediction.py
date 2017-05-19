import numpy as np
import read_data as reader
from sklearn.externals import joblib
from read_data import  root

def normalize(image):
    image = np.asanyarray(image)
    image = image.reshape(10000, -10000)
    return image

def main():
    test_images_path = root + "t10k-images-idx3-ubyte.gz"
    test_labels_path = root + "t10k-labels-idx1-ubyte.gz"
    test_labels = reader.get_labels(test_labels_path)
    test_images = reader.get_grayscaled_images(test_images_path)

    print("Loading classifier...")
    classifier_path = root + "digits_classifier.joblib.pkl"
    classifier = joblib.load(classifier_path)

    i = 0
    match = 0
    print("Predicting test samples classes...")
    test_images = normalize(test_images)
    result = classifier.predict(test_images)

    while(i < 10000):
        if(result[i] == test_labels[i]):
            match += 1
        i += 1

    print("Matched: ", match)

if  __name__ =='__main__':main()