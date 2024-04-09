# Iris-Logistic-Classification
Iris Logistic Classification


Overview:
This repository contains a Python implementation of binary logistic regression aimed at classifying iris species. Specifically, it differentiates Setosa from the other species within the Iris dataset. Utilizing gradient descent for optimization, the script provides a straightforward example of logistic regression applied to a well-known dataset.

Features:
Binary logistic regression model
Use of gradient descent for optimization
Evaluation of model performance with accuracy and a confusion matrix
Visualization of model predictions

Requirements:
Python 3
NumPy (for numerical computations)
Scikit-learn (for metrics and dataset handling)
Matplotlib (for plotting graphs)
These dependencies can be installed using pip:

"pip install numpy scikit-learn matplotlib"

Dataset:
The program uses the iris.data file, which should be present in the same directory as the script. This dataset is famously used for various machine learning tasks and includes 150 instances of iris plants, categorized into three species.

Usage Instructions:
Ensure all required libraries mentioned above are installed.
Clone this repository or download the irisClassifier.py, sigmoid.py, and iris.data files to the same directory.
Run the script from the terminal:
"python irisClassifier.py"
Upon execution, the script trains a logistic regression model to classify the Iris Setosa species against the others and outputs the model's performance metrics.

License:
This project is released under the MIT License.

