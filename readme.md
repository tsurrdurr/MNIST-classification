# MNIST-classification

An attempt in exploring scikit-learn functionality and machine learning algorithms using samples provided by MNIST database.

## Installation

I recomend to use [virtualenv](https://github.com/pypa/virtualenv) and set up a new environment for this project.

Python version used is **Python 3.6**.

Linux installation:   
You may simply run the following code to install all the dependencies:     
`pip install -r requirements.txt`   

Windows installation:  
With Windows, installation from _requirements.txt_ doesn't quite work out for me, but [this link](http://www.lfd.uci.edu/~gohlke/pythonlibs/) helped me out a lot with Windows wheels. 
Download the following packages for your system and python 3.6:  
[numpy+mkl](http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy)  
[scikit-learn](http://www.lfd.uci.edu/~gohlke/pythonlibs/#scikit-learn)  
[scipy](http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy)  
and install them with:  
`pip install .\wheel_name.whl`.  

## Running

Download 4 MNIST database files from [here](http://yann.lecun.com/exdb/mnist/) and put them all to a folder where temporary files may be stored. **Do not** extract archives content.  
Set `root` variable in `read_data.py` to folder where you have put the MNIST samples.  
You would want to run `read_data.py`, then `classification.py` with desired parameter and finally `prediction.py`.

## read_data.py

Reads the training data in a format, described [at the bottom of official MNIST page](http://yann.lecun.com/exdb/mnist/). 

The resulting arrays containing images and their labels are saved with scikit-learn tool `joblib` (so you only have to do it once).

## classification.py

Creates a classifier — object capable of determining a class of input data object. 

List of `classification.py` parameters:  
`-svc` - use [Linear SVC classifier](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)  
`-sgd` - use [SGD classifier](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)  
`-nb` - use [Multinomial Naive Bayes classifier](http://scikit-learn.org/stable/modules/naive_bayes.html#multinomial-naive-bayes)  
`-kn` - use [KNeighbors classifier](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

Arrays from earlier are read from disk and reshaped with `numpy` tools. Then a classifier of selected type is created based on input images and their classes (0-9). 
After this accuracy of the classifier is displayed (percentage of correctly classified values from the training set).  
Classifier is also saved to disk.


## prediction.py

Tests the generated classifier with test data. 

Classifier from earlier read from disk and test values are parsed according to MNIST format description. `predict` function of the classifier returns array of predicted values of test images. Then these predictions are compared to test labels.  

For now ~91.4% of test values are recognized with Linear SVC.  
KNeighbors prediction test result shows 96.5% accuracy, but it takes really much time and disk space to create a classifier.