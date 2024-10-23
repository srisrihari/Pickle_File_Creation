# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle

# 1. Load and inspect the dataset
heart_data = pd.read_csv('heart.csv')  # Load heart disease dataset
print(heart_data.head())  # Display the first few rows of the dataset

# 2. Preprocessing: Encoding categorical variables and scaling continuous variables
heart_data_x = heart_data.drop("HeartDisease", axis=1)  # Features (input variables)
heart_data_y = heart_data['HeartDisease']  # Target variable (output)

# One-hot encode categorical variables (converts categorical variables into a format that can be provided to ML algorithms)
heart_data_x_encoded = pd.get_dummies(heart_data_x, drop_first=True).astype(int)

# Standard scaling for continuous features to normalize the data
continuous_features = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
scaler = StandardScaler()  # Initialize the scaler
heart_data_x_encoded[continuous_features] = scaler.fit_transform(heart_data_x_encoded[continuous_features])  # Scale the features

# 3. Splitting data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(heart_data_x_encoded, heart_data_y, test_size=0.3, random_state=42)

# 4. Define and train the Logistic Regression model
logreg_model = LogisticRegression(max_iter=1000)  # Create a logistic regression model with maximum iterations set to 1000
logreg_model.fit(X_train, y_train)  # Train the model using the training data

# 5. Save the model and scaler into a pickle file
with open('heart_disease_logreg_model.pkl', 'wb') as file:  # Open a file to write the pickle
    pickle.dump({
        'logistic_regression': logreg_model,  # Save the trained model
        'scaler': scaler,  # Save the scaler
        'features': heart_data_x_encoded.columns.tolist()  # Save the feature names for reference
    }, file)  # Save all in a dictionary format

# At this point, the model and scaler are saved into 'heart_disease_logreg_model.pkl'
