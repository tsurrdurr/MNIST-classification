import read_data as reader
from sklearn.externals import joblib

def main():
    test_images_path = "C:\\Work\\nn\\t10k-images-idx3-ubyte.gz"
    test_labels_path = "C:\\Work\\nn\\t10k-labels-idx1-ubyte.gz"
    test_labels = reader.get_labels(test_labels_path)
    test_images = reader.get_grayscaled_images(test_images_path)

    classifier_path = "C:\\temp\\digits_classifier.joblib.pkl"
    classifier = joblib.load(classifier_path)

    i = 0
    match = 0
    test_images = normalize(test_images)
    result = classifier.predict(test_images)

    while(i < 10000):
        if(result[i] == test_labels[i]):
            match += 1
        i += 1

    print("matched: ", match)

if  __name__ =='__main__':main()