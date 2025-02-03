import pandas as pd
import requests
from bs4 import BeautifulSoup
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import re
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

# URL of the Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Fetch the raw text
response = requests.get(url)

if response.status_code != 200:
    raise Exception(f"Failed to fetch webpage. Status code: {response.status_code}")

# Parse with BeautifulSoup
soup = BeautifulSoup(response.text, "html.parser")

# Extract the raw text
raw_text = soup.get_text()  # Gets the text content of the page

# Split by lines and clean up
data = [line.strip().split(",") for line in raw_text.split("\n") if line]

# Define column names (since the dataset lacks headers)
columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

# Create DataFrame
df = pd.DataFrame(data, columns=columns)

# Display first few rows
print(df.head())

columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

#typecasting and normalizing and cleaning

lower_quantile = 0.05  # Lower 1st percentile
upper_quantile = 0.95  # Upper 99th percentile

# Step 1: Loop through each column in the 'columns' list
for x in columns:
    # Step 2: Typecast the column to float
    df[x] = df[x].astype(float)
    #if x != "petal_width":
    # Step 3: Calculate quantile ranges
    q_low = df[x].quantile(lower_quantile)
    q_high = df[x].quantile(upper_quantile)

    # Step 4: Remove outliers outside the quantile range
    df = df[(df[x] >= q_low) & (df[x] <= q_high)]
    
    # Step 5: Normalize the column to [0, 1]
    df[x] = (df[x] - df[x].min()) / (df[x].max() - df[x].min())

print("\n")
print(df.head())
print("\n")


#look at how u can do the following with the data you scraped
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print("Feature Statistiken:")
print(X.describe())

print("\nTarget-Verteilung:")
print(y.value_counts()) #change this since this function is probably not in pandas idea count uniques and show us what has what type things

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y) 

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42) 


#creating the training and test datasets
X_train_df = pd.DataFrame(X_train)
X_train_df.to_csv('iris_training_values.csv', index=False)
X_test_df = pd.DataFrame(X_test)
X_test_df.to_csv('iris_testing_values.csv', index=False)

#getting the first row to be used in the activation of the OLS and the ANN model 
activation = df.sample(n=1)
print(activation)
activation.to_csv('iris_activation.csv', index=False)

# ... [vorhandener Code bis Zeile 70] ...

### Subgoal 4: TensorFlow AI-Modell ###
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# One-Hot-Encoding der Zielvariable
y_train_onehot = to_categorical(y_train, num_classes=3)
y_test_onehot = to_categorical(y_test, num_classes=3)

# Modellarchitektur
model = Sequential([
    Dense(64, activation='relu', input_shape=(4,)),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])

# Kompilieren und Training
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train_onehot,
    epochs=50,
    batch_size=16,
    validation_data=(X_test, y_test_onehot)
)

# Modell speichern
model.save('currentAISolution.h5')

# Performance-Visualisierung
plt.figure(figsize=(12, 4))

# Trainingsverlauf
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curve')
plt.legend()
plt.savefig('learningBase/training_performance.png')
plt.close()

# -------------------- Subgoal 5: OLS Model (Statsmodels) --------------------

# Add constant for OLS regression
X_train_ols = sm.add_constant(X_train)
X_test_ols = sm.add_constant(X_test)

# Fit OLS model
ols_model = sm.OLS(y_train, X_train_ols).fit()

# Save the model summary
with open('currentOlsSolution.txt', 'w') as f:
    f.write(ols_model.summary().as_text())

# Save the OLS model
with open('currentOlsSolution.pkl', 'wb') as f:
    pickle.dump(ols_model, f)

# Predictions and Evaluation
y_pred_ols = ols_model.predict(X_test_ols).round().astype(int)

print("\nOLS Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_ols):.2f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_ols))

# Visualizations
plt.figure(figsize=(12, 4))

# Scatter Plot
plt.subplot(1, 2, 1)
sns.scatterplot(x=y_test, y=y_pred_ols, hue=y_test, palette='viridis')
plt.title('Actual vs Predicted (OLS)')
plt.xlabel('True Class')
plt.ylabel('Predicted Class')

# Residual Distribution
plt.subplot(1, 2, 2)
residuals = y_test - y_pred_ols
sns.histplot(residuals, kde=True)
plt.title('Residual Distribution')
plt.savefig('learningBase/ols_performance.png')
plt.close()

# Pairplot for combined dataset
combined_data = pd.concat([pd.DataFrame(X, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']), pd.Series(y, name='species')], axis=1)
sns.pairplot(combined_data, hue='species', diag_kind='kde')
plt.suptitle("Pairplot der Iris-Daten", y=1.02)
plt.savefig("iris_pairplot.pdf")
plt.show()

# Boxplot for each feature
plt.figure(figsize=(10, 8))
sns.boxplot(data=combined_data.drop('species', axis=1))
plt.title("Boxplot der Merkmale")
plt.savefig("iris_boxplot.pdf")
plt.show()
