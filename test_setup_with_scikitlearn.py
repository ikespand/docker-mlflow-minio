# -*- coding: utf-8 -*-
"""
A sample progrma to test if our MLFlow setup is working. 

The main program was shamelessly copied from official docs:
    https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html

Following were the original authors and license.
    Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
    License: BSD 3 clause
"""

import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import mlflow
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error
from urllib.parse import urlparse
logging.basicConfig(level=logging.WARN)

# Defining MLFlow URI
URI = r"http://localhost:5000"
mlflow.set_tracking_uri(URI)
mlflow.set_experiment("MyMLTask")
# %% Load the dataset do the preprocesisng and split
digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.3, shuffle=False)

# %% Perform the training and log everything to mlflow
with mlflow.start_run() as run:
    # Create a classifier: a support vector classifier
    gamma = 0.001
    kernel="rbf"
    clf = svm.SVC(gamma=gamma, kernel=kernel)
    # Learn the digits on the train subset
    clf.fit(X_train, y_train)
    # Predict the value of the digit on the test subset
    y_pred = clf.predict(X_test)
    
    # Some visualization
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, y_pred):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")
        
    print(
        f"Classification report for classifier {clf}:\n"
        f"{metrics.classification_report(y_test, y_pred)}\n"
    )    
    
    # Some metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    # Log parameters with which we want to experiment and record results
    mlflow.log_param("gamma", gamma) 
    mlflow.log_param("kernel", kernel)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    # Log figure to visualize after the runs
    mlflow.log_figure(fig, 'comparision.png')
    # Save the model
    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
    model_signature = mlflow.models.signature.infer_signature(X_train, y_train)
    run_id = run.info.run_uuid
    experiment_id = run.info.experiment_id
    if tracking_url_type_store != "file":
        mlflow.sklearn.log_model(clf, "clf")
    else:
        mlflow.sklearn.log_model(clf, "clf", signature=model_signature)
