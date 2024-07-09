# Install seaborn using the %pip magic command

# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
df = sns.load_dataset('iris')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Display basic information and summary statistics
print("\nBasic Information:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

# Histograms
print("\nHistograms:")
df.hist(figsize=(10, 8), bins=20)
plt.show()

# Boxplots
print("\nBoxplots:")
plt.figure(figsize=(10, 8))
sns.boxplot(data=df)
plt.show()

# Pair plot
print("\nPair Plot:")
sns.pairplot(df, hue='species')
plt.show()

# Correlation heatmap
print("\nCorrelation Heatmap:")
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.show()

# Violin plots
print("\nViolin Plot:")
plt.figure(figsize=(10, 8))
sns.violinplot(x='species', y='sepal_length', data=df)
plt.show()

# Scatter plots with linear regression lines
print("\nScatter Plot with Linear Regression Lines:")
sns.lmplot(x='sepal_length', y='sepal_width', data=df, hue='species', height=6)
plt.show()
