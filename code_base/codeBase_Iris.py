import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import models
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Define file paths
model_path_OLS = "./knowledge_base/currentOlsSolution.pkl"
model_path_ANN = "./knowledge_base/currentAISolution.h5"
data_path = "./activation_base/iris_activation.csv"

# Check if files exist
if not os.path.exists(model_path_ANN):
    raise FileNotFoundError(f"Model file not found at {model_path_ANN}")
if not os.path.exists(model_path_OLS):
    raise FileNotFoundError(f"Model file not found at {model_path_OLS}")
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data file not found at {data_path}")

# Load the dataset
data = pd.read_csv(data_path)

# Ensure correct feature columns
feature_columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
if not set(feature_columns).issubset(data.columns):
    raise ValueError(f"Expected columns {feature_columns} missing in dataset.")

# Ensure target variable is properly encoded
if "class" not in data.columns:
    raise ValueError("Expected 'class' column missing in dataset.")

# Load the LabelEncoder used during training
encoder = LabelEncoder()
encoder.classes_ = np.array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])  # Ensure correct mapping

# Convert target variable to encoded integers
data["class"] = encoder.transform(data["class"])

# Split dataset
datasplitter = int(len(data) * 0.8)
test_data = data.iloc[datasplitter:]

# Prepare features and labels
X_test = test_data[feature_columns].values
y_test = test_data["class"].values

# Load ANN model
model = tf.keras.models.load_model(model_path_ANN)

# Convert y_test to one-hot encoding for categorical crossentropy evaluation
num_classes = len(encoder.classes_)
y_test_onehot = to_categorical(y_test, num_classes=num_classes)

# Predict using ANN
y_pred = model.predict(X_test)

# Evaluate ANN model
loss, accuracy = model.evaluate(X_test, y_test_onehot)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Plot ANN predictions
plt.figure(figsize=(10, 8))
plt.scatter(range(len(y_test)), y_test, label='True Labels', color='blue')
plt.scatter(range(len(y_pred)), np.argmax(y_pred, axis=1), label='Predicted by ANN', color='red', marker='x')
plt.xlabel('Sample Index')
plt.ylabel('Class')
plt.title('Actual vs. Predicted Class (ANN Model)')
plt.legend()
plt.tight_layout()
plt.savefig("./output/ai_model_plot.png")
plt.show()
