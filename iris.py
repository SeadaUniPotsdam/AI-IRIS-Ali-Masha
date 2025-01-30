import pandas as pd
import requests
from bs4 import BeautifulSoup

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
