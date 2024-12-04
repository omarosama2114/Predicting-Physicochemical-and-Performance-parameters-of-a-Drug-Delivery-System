import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from scipy.stats import chi2_contingency

# Load the data
df = pd.read_csv('C:/Users/omaro/Desktop/KAU/Drug Dataset - Formatted.csv')

# Drop Unnamed Columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Remove leading/trailing spaces in column names
df.columns = df.columns.str.strip()

numeric_features = df[df.select_dtypes(include=[np.number]).columns]

# Calculate the correlation matrix
correlation_matrix = numeric_features.corr()

# Plot the correlation matrix
plt.figure(figsize=(26, 20))  

# Create heatmap
sns.heatmap(correlation_matrix, cmap='Reds', annot=True, fmt='.2f', cbar=True)

plt.xticks(rotation=45, ha='right', fontsize=12)  
plt.yticks(fontsize=12)                          
plt.title('Feature Correlation Heatmap', fontsize=20)

plt.show()

