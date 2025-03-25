import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Create directories if they don't exist
os.makedirs('Data', exist_ok=True)
os.makedirs('Models', exist_ok=True)

# Load data
data = pd.read_csv('Data/Loan_Data.csv')

# Comprehensive Descriptive Analysis
print("Dataset Information:")
print(data.info())

# Detailed statistics
desc_stats = data.describe()
print("\nDescriptive Statistics:")
print(desc_stats)

# Class Distribution
print("\nClass Distribution:")
class_dist = data['default'].value_counts(normalize=True)
print(class_dist)

# Visualize Class Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='default', data=data)
plt.title('Distribution of Default vs Non-Default')
plt.xlabel('Default Status')
plt.ylabel('Count')
plt.savefig('Data/default_distribution.png')
plt.close()

# Correlation Analysis
plt.figure(figsize=(12, 10))
correlation_matrix = data.drop('customer_id', axis=1).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Features')
plt.tight_layout()
plt.savefig('Data/correlation_matrix.png')
plt.close()

# Feature Distribution by Default Status
features_to_plot = ['credit_lines_outstanding', 'loan_amt_outstanding', 'total_debt_outstanding',
                    'income', 'years_employed', 'fico_score']

plt.figure(figsize=(15, 10))
for i, feature in enumerate(features_to_plot, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x='default', y=feature, data=data)
    plt.title(f'{feature} by Default Status')
plt.tight_layout()
plt.savefig('Data/features_by_default.png')
plt.close()

# Prepare data for model training
X = data.drop(["default", "customer_id"], axis=1)
y = data["default"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Additional class distribution checks
print("\nTraining Set Class Distribution:")
print(y_train.value_counts(normalize=True))

print("\nTest Set Class Distribution:")
print(y_test.value_counts(normalize=True))

# Save processed data with scaling information
processed_data = pd.DataFrame(X_scaled, columns=X.columns)
processed_data['default'] = y
processed_data.to_csv('Data/Processed_Loan_Data.csv', index=False)

# Save scaler for later use
import joblib
joblib.dump(scaler, 'Models/feature_scaler.joblib')

print("Preprocessing completed. Detailed analysis saved in the Data and Models directories.")