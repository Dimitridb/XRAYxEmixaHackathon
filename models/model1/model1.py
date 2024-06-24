# model1.py

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target
X = X.astype(np.float32)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Train the model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Create directory if it does not exist
os.makedirs('models/model1', exist_ok=True)

# Save the trained model to a file
with open('models/model1/model1.pkl', 'wb') as f:
    pickle.dump(clf, f)
