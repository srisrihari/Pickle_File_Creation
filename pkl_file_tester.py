# ========================================
# Testing the saved model with new data
# ========================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

# Define continuous features (same as used in training)
continuous_features = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]

# 1. Create a new sample dataset (mimicking the structure of the original dataset)
new_data = {
    'Age': [45, 60, 72],               # Continuous feature
    'Sex': ['M', 'F', 'M'],            # Categorical feature (will be one-hot encoded)
    'ChestPainType': ['ATA', 'NAP', 'ASY'], # Categorical feature
    'RestingBP': [120, 140, 130],      # Continuous feature
    'Cholesterol': [240, 200, 280],    # Continuous feature
    'FastingBS': [0, 1, 0],            # Binary feature
    'RestingECG': ['Normal', 'ST', 'LVH'], # Categorical feature
    'MaxHR': [150, 130, 100],          # Continuous feature
    'ExerciseAngina': ['N', 'Y', 'Y'], # Categorical feature
    'Oldpeak': [1.0, 2.3, 3.5],        # Continuous feature
    'ST_Slope': ['Up', 'Flat', 'Down'] # Categorical feature
}

# Convert the new data to a DataFrame
new_data_df = pd.DataFrame(new_data)  # Create a DataFrame from the new data

# 2. Load the saved model and scaler from the pickle file
with open('heart_disease_logreg_model.pkl', 'rb') as file:  # Open the pickle file for reading
    saved_objects = pickle.load(file)  # Load the contents of the pickle file

# Access the scaler and model from the loaded objects
scaler = saved_objects['scaler']  # Retrieve the scaler
logreg_model = saved_objects['logistic_regression']  # Retrieve the trained logistic regression model
model_features = saved_objects['features']  # Load the original feature names

# 3. Preprocess the new data (one-hot encoding and scaling)
# One-hot encode the categorical variables (ensure the same structure as the original)
new_data_encoded = pd.get_dummies(new_data_df, drop_first=True).astype(int)  # One-hot encoding

# 4. Scale the continuous features in the new dataset using the saved scaler
new_data_encoded[continuous_features] = scaler.transform(new_data_encoded[continuous_features])  # Scale the features

# 5. Ensure that the new dataset has the same columns as the training data
for col in model_features:  # Loop through the original feature names
    if col not in new_data_encoded.columns:  # Check if the column is missing
        new_data_encoded[col] = 0  # Add missing columns with default value 0

new_data_encoded = new_data_encoded[model_features]  # Reorder columns to match the training set

# 6. Make predictions using the loaded Logistic Regression model
logreg_predictions = logreg_model.predict(new_data_encoded)  # Make predictions

# Output the predictions
print("Logistic Regression Predictions:", logreg_predictions)  # Display the predictions
