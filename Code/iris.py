import pandas as pd
import requests
from bs4 import BeautifulSoup
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import pickle  # For saving the OLS model
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix  # For evaluating the OLS model

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# URL of the Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Fetch raw data
response = requests.get(url)

if response.status_code != 200:
    raise Exception(f"Error fetching the webpage. Status code: {response.status_code}")

# Parse HTML with BeautifulSoup
soup = BeautifulSoup(response.text, "html.parser")

# Extract raw data
raw_text = soup.get_text()  # Text content of the page

# Split data into lines and clean
data = [line.strip().split(",") for line in raw_text.split("\n") if line and len(line.strip().split(",")) == 5]

# Define column names (since the dataset has no header)
columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

# Create DataFrame
df = pd.DataFrame(data, columns=columns)

# Display first rows
print("First rows of the dataset:")
print(df.head())

# Select columns for processing
numeric_columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# Typecasting, normalization, and outlier removal
lower_quantile = 0.05  # Lower 5th percentile
upper_quantile = 0.95  # Upper 95th percentile

for col in numeric_columns:
    # Convert column to float
    df[col] = df[col].astype(float)

    # Calculate quantiles
    q_low = df[col].quantile(lower_quantile)
    q_high = df[col].quantile(upper_quantile)

    # Remove outliers
    df = df[(df[col] >= q_low) & (df[col] <= q_high)]

    # Normalize to [0, 1]
    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

print("\nCleaned and normalized dataset:")
print(df.head())

# Split features (X) and target variable (y)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Display feature statistics
print("\nFeature statistics:")
print(X.describe())

# Display target variable distribution
print("\nTarget variable distribution:")
print(y.value_counts())

# Encode target variable
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Save training and test data
X_train_df = pd.DataFrame(X_train, columns=numeric_columns)
X_train_df.to_csv('./learning_base/iris_training_values.csv', index=False)
X_test_df = pd.DataFrame(X_test, columns=numeric_columns)
X_test_df.to_csv('./learning_base/iris_testing_values.csv', index=False)

y_train_df = pd.DataFrame(y_train, columns=["class"])
y_train_df.to_csv('./learning_base/iris_training_target.csv', index=False)
y_test_df = pd.DataFrame(y_test, columns=["class"])
y_test_df.to_csv('./learning_base/iris_testing_target.csv', index=False)

# Select a random row for activation
activation = df.sample(n=1)
print("\nActivation data point:")
print(activation)
activation.to_csv('./activation_base/iris_activation.csv', index=False)

# -------------------- AI Model (Neural Network) --------------------

# One-hot encoding for the target variable
num_classes = len(np.unique(y_train))
y_train_onehot = to_categorical(y_train, num_classes=num_classes)
y_test_onehot = to_categorical(y_test, num_classes=num_classes)

# Model architecture
model = Sequential([
    Dense(64, activation='relu', input_shape=(4,)),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    X_train, y_train_onehot,
    epochs=50,
    batch_size=16,
    validation_data=(X_test, y_test_onehot)
)

# Save model
model.save('./learning_base/currentAISolution.h5')

# Performance visualization
plt.figure(figsize=(12, 4))

# Training history (Loss)
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.legend()

# Training history (Accuracy)
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curve')
plt.legend()

plt.savefig('./learning_base/training_performance.png')
plt.close()

# -------------------- OLS Model (Statsmodels) --------------------

import statsmodels.api as sm

# Add constant for OLS regression
X_train_ols = sm.add_constant(X_train)
X_test_ols = sm.add_constant(X_test)

# Fit OLS model
ols_model = sm.OLS(y_train, X_train_ols).fit()

# Save model summary
with open('./learning_base/currentOlsSolution.txt', 'w') as f:
    f.write(ols_model.summary().as_text())

# Save OLS model
with open('./learning_base/currentOlsSolution.pkl', 'wb') as f:
    pickle.dump(ols_model, f)

# Predictions and evaluation
y_pred_ols = ols_model.predict(X_test_ols).round().astype(int)

print("\nOLS Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_ols):.2f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_ols))

# Visualizations
plt.figure(figsize=(12, 4))

# Scatter plot (Actual vs. Predicted values)
plt.subplot(1, 2, 1)
sns.scatterplot(x=y_test, y=y_pred_ols, hue=y_test, palette='viridis')
plt.title('Actual vs. Predicted (OLS)')
plt.xlabel('True Class')
plt.ylabel('Predicted Class')

# Residual distribution
plt.subplot(1, 2, 2)
residuals = y_test - y_pred_ols
sns.histplot(residuals, kde=True)
plt.title('Residual Distribution')

plt.savefig('./learning_base/ols_performance.png')
plt.close()

# Pairplot for the entire dataset
combined_data = pd.concat([X, pd.Series(y, name='species')], axis=1)
sns.pairplot(combined_data, hue='species', diag_kind='kde')
plt.suptitle("Pairplot of Iris Data", y=1.02)
plt.savefig("./learning_base/iris_pairplot.pdf")
plt.show()

# Boxplot for each feature
plt.figure(figsize=(10, 8))
sns.boxplot(data=combined_data.drop('species', axis=1))
plt.title("Boxplot of Features")
plt.savefig("./learning_base/iris_boxplot.pdf")
plt.show()