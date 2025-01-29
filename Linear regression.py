# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample dataset (for illustration purposes)
data = {
    'square_footage': [1500, 2000, 2500, 3000, 3500, 4000, 4500],
    'bedrooms': [3, 4, 4, 5, 3, 4, 5],
    'bathrooms': [2, 3, 3, 4, 2, 3, 4],
    'price': [300000, 450000, 500000, 600000, 350000, 500000, 650000]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Features and target variable
X = df[['square_footage', 'bedrooms', 'bathrooms']]
y = df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict prices
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Display the model coefficients
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
