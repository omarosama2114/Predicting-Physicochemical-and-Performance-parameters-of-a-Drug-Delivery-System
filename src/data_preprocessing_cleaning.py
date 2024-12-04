import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from scipy.stats import chi2_contingency

# Load the data
df = pd.read_csv('../Data/Raw Drugs Dataset.csv')

# Drop Unnamed Columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Remove leading/trailing spaces in column names
df.columns = df.columns.str.strip()

# Define label columns
labels = [
    "Drug loading (EE%)", 
    "Vesicular  Size (nm)", 
    "Vesicles zeta potential (mV)", 
    "Drug released percentage"
]

# Change 'Drug MW' type to Numeric
df['Drug MW (Dalton)'] = pd.to_numeric(df['Drug MW (Dalton)'], errors='coerce')

# Select numeric columns excluding labels
numeric_feature_cols = df.select_dtypes(include=[np.number]).columns.difference(labels)

# Initialize the imputer 'Mean Strategy'
imputer = SimpleImputer(strategy='mean')

# Replace missing values in numeric columns with the mean
df.loc[:, numeric_feature_cols] = imputer.fit_transform(df[numeric_feature_cols])

# Round the numeric columns to 2 decimal places
df.loc[:, numeric_feature_cols] = df[numeric_feature_cols].round(2)

# Replace missing values in non-numeric columns with 'Unknown'
non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
df.loc[:, non_numeric_cols] = df[non_numeric_cols].fillna("Unknown")

# Drop Duplicate Rows
df = df.drop_duplicates()

# Drop Columns with Strong Correlation with other columns
# Drug MW and Drug (Log P) have a strong correlation 
df = df.drop(columns=['DRUG (Log P)'])

# Encode Categorical Columns 'Phosopholipid Type' using One-Hot Encoding
# Perform one-hot encoding
df = pd.get_dummies(df, columns=['Phosopholipid Type'], prefix='PL', drop_first=True)










