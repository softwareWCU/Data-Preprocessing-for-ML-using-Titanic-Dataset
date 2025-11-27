import pandas as pd

# Load dataset (replace with your file name)
df = pd.read_csv("titanic.csv")

# Display dataset details
print("Dataset Shape:", df.shape)
print("\nDataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe(include='all'))
