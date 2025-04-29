# Task: Analyze Data with Pandas and Visualize Results with Matplotlib

# --- Import Required Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# --- Load the Dataset ---
try:
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    print("Dataset loaded successfully!\n")
except Exception as e:
    print(f"Error loading dataset: {e}")

# --- Explore the Dataset ---
print("First 5 rows:")
print(df.head())

print("\nData Types and Missing Values:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

# --- Clean Dataset ---
# No missing values in this dataset; otherwise, use df.dropna() or df.fillna()

# --- Task 2: Basic Data Analysis ---
print("\nDescriptive Statistics:")
print(df.describe())

# Grouping
group_mean = df.groupby('species').mean()
print("\nMean values per species:")
print(group_mean)

# Observation example:
print("\nObservation:")
print("Setosa species has significantly smaller petal length and width than the other species.")

# --- Task 3: Data Visualization ---
sns.set(style="whitegrid")  # Optional: Use seaborn style

# 1. Line Chart - simulate time-series using index
plt.figure(figsize=(10,5))
plt.plot(df.index, df['sepal length (cm)'], label='Sepal Length')
plt.title("Sepal Length over Entries (as time)")
plt.xlabel("Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.tight_layout()
plt.show()

# 2. Bar Chart - average petal length per species
plt.figure(figsize=(7,5))
group_mean['petal length (cm)'].plot(kind='bar', color='skyblue')
plt.title("Average Petal Length by Species")
plt.ylabel("Petal Length (cm)")
plt.xlabel("Species")
plt.tight_layout()
plt.show()

# 3. Histogram - petal width distribution
plt.figure(figsize=(7,5))
plt.hist(df['petal width (cm)'], bins=15, color='coral', edgecolor='black')
plt.title("Distribution of Petal Width")
plt.xlabel("Petal Width (cm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 4. Scatter Plot - sepal length vs. petal length
plt.figure(figsize=(7,5))
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species')
plt.title("Sepal Length vs. Petal Length by Species")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.tight_layout()
plt.show()
