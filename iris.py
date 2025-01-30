import pandas as pd
import requests
from bs4 import BeautifulSoup
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import re
import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler

O
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

print(df.head())
