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

y_train_df = pd.DataFrame(y_train)
y_train_df.to_csv('iris_training_target.csv', index=False)
y_test_df = pd.DataFrame(y_test)
y_test_df.to_csv('iris_testing_target.csv', index=False)

#getting the first row to be used in the activation of the OLS and the ANN model 
activation = df.sample(n=1)
print(activation)
activation.to_csv('iris_activation.csv', index=False)
