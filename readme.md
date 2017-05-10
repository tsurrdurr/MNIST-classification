# MNIST-classification

An attempt in exploring scikit-learn functionality and machine learning algorithms with MNIST database.

## Installation

Download and install `scikit-learn`, `scipy`, `numpy` packages. Dependencies are complicated, but [this link](http://www.lfd.uci.edu/~gohlke/pythonlibs/) helped me out a lot.

## Running

Download 4 MNIST database files from [here](http://yann.lecun.com/exdb/mnist/) and put them all to folder where temporary files may be stored. Do NOT extract archives content.
Set `root` variable in `read_data.py` to folder where you put the MNIST samples.
Run `read_data.py`, then `classification.py` and finally `prediction.py`.

## read_data.py

Reads the data in a format, described [at the bottom of official MNIST page](http://yann.lecun.com/exdb/mnist/). The resulting arrays are saved with scikit-learn tool `joblib`.

## classification.py

Arrays from earlier are read from disk and reshaped with `numpy` tools. Then a LinearSVC classificator is created based on input images and their classes (0-9). Classificator is also saved to disk.

## prediction.py

Classificator from earlier read from disk and test values are parsed according to MNIST format description. `predict` function of the classifier returns array of predicted values of test images. Then these predictions are compared to test labels.  

For now ~91.4% of test values are recognized. May implement other algorithms in time.