import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Create directories if they don’t exist
os.makedirs('Data', exist_ok=True)
os.makedirs('Models', exist_ok=True)

# Load data
data = pd.read_csv('Data/Loan_Data.csv')

# Descriptive statistics
data.describe().to_csv('Data/Loan_Data_Describe.csv')

# Generate static plots (optional, excluded from Docker)
plt.figure(figsize=(8, 6))
sns.countplot(x='default', data=data)
plt.savefig('Data/default_distribution.png')
plt.close()

plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.savefig('Data/correlation_matrix.png')
plt.close()

# Save processed data
data.to_csv('Data/Processed_Loan_Data.csv', index=False)

print("Preprocessing completed. Data saved to Data/Processed_Loan_Data.csv and Data/Loan_Data_Describe.csv")
