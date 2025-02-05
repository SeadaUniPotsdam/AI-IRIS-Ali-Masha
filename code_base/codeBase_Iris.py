import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

# Define file paths
model_path_OLS = "/local_model/currentOLsSolution.xml"
model_path_ANN = "/local_model/currentAISolution.h5"
data_path = "/local_data/currentActivation.csv"

if not os.path.exists(model_path_ANN):
    raise FileNotFoundError(f"Model file not found at {model_path_ANN}")
if not os.path.exists(model_path_OLS):
    raise FileNotFoundError(f"Model file not found at {model_path_OLS}")
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data file not found at {data_path}")

data = pd.read_csv(data_path)
model = tf.keras.models.load_model(model_path_ANN)

datasplitter = int(len(data) * .8)
test_data = data[datasplitter:]
X_test = test_data.get("x")
y_test = test_data.get("y")

# Predict using the trained model
y_pred = model.predict(X_test)

# Step 7: Evaluate the model (optional)
loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")

# Create plot
plt.figure(figsize=(10, 8))
plt.scatter(X_test, y_test, label='Data Points', color='blue')
plt.scatter(X_test, y_pred, label='Predicted by Neural Network', color='red', linestyle='--')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Function Approximation with Neural Network')

plt.legend()
plt.tight_layout()
plt.savefig("/output/ai_model_plot.png")

# Parse XML and extract model parameters
tree = ET.parse(model_path_OLS)
root = tree.getroot()

# Extract coefficients (assuming they are stored in <coefficients> and <intercept> tags)
intercept = float(root.find("intercept").text)
coefficients = [float(coef.text) for coef in root.findall("coefficients/coef")]

# Prepare test data
datasplitter = int(len(data) * .8)
test_data = data[datasplitter:]
X_test = test_data[["x"]].values  # Convert to numpy array if needed
y_test = test_data["y"]

# Compute predictions manually: y = intercept + coef * x
y_pred = intercept + coefficients[0] * X_test.flatten()

# Plot the results
plt.figure(figsize=(10, 8))
plt.scatter(X_test, y_test, label='Data Points', color='blue')
plt.scatter(X_test, y_pred, label='Predicted by OLS', color='red', marker='x')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Function Approximation with OLS Model')

plt.legend()
plt.tight_layout()
plt.savefig("/output/ols_model_plot.png")
