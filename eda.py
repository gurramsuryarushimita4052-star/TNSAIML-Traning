import pandas as pd

print("Starting EDA...")

# Load the dataset
df = pd.read_csv("dataset/Housing.csv")

# Display first five rows
print("\nFIRST FIVE ROWS")
print(df.head())

# Display rows and columns
print("\nDATASET SHAPE")
print(df.shape)

# Display column names
print("\nCOLUMN NAMES")
print(df.columns.tolist())

# Display missing values
print("\nMISSING VALUES")
print(df.isnull().sum())

# Display duplicate rows
print("\nDUPLICATE ROWS")
print(df.duplicated().sum())

# Display statistics
print("\nSTATISTICAL SUMMARY")
print(df.describe())

print("\nEDA COMPLETED SUCCESSFULLY!")