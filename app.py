# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 2: Load Dataset
df = pd.read_csv("kelal.csv")
print(df.head())
print(df.info())
print(df.describe())

# Step 3: Basic Data Cleaning
df = df.dropna()

# Step 4: (Skip – Target already exists)

# Step 5: Data Visualization
sns.countplot(x="Target", data=df)
plt.title("Target Class Distribution")
plt.show()

# Correlation Heatmap (only works on numeric columns)
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()

# Step 6: Encode Categorical Variables
df_encoded = pd.get_dummies(df, drop_first=True)

# Step 7: Feature & Target Split
X = df_encoded.drop("Target", axis=1)
y = df_encoded["Target"]

# Step 8: Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 9: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 10: Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 11: Evaluate the Model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
