import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load data
df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

# Basic preprocessing
def preprocess_data(df):
    # Create copy
    data = df.copy()
    
    # Fill missing values
    data['Age'].fillna(data['Age'].median(), inplace=True)
    
    # Convert gender to numeric
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    
    # Select features
    X = data[['Sex', 'Age', 'Pclass']]
    y = data['Survived']
    
    return X, y

# Preprocess data
X, y = preprocess_data(df)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save model and scaler
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Print accuracy
print(f"Model accuracy: {model.score(X_test_scaled, y_test):.2f}")