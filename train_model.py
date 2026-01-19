import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

# Create sample data for house price prediction
np.random.seed(42)
n_samples = 1000

# Generate synthetic data
areas = np.random.uniform(500, 5000, n_samples)  # Area in sq ft
bedrooms = np.random.randint(1, 6, n_samples)   # Number of bedrooms
bathrooms = np.random.randint(1, 4, n_samples)  # Number of bathrooms

# Create price based on features with some noise
prices = (
    areas * 200 +                    # Price per sq ft
    bedrooms * 10000 +              # Additional price per bedroom
    bathrooms * 5000 +              # Additional price per bathroom
    np.random.normal(0, 10000, n_samples)  # Noise
)

# Create DataFrame
df = pd.DataFrame({
    'area': areas,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'price': prices
})

# Prepare features and target
X = df[['area', 'bedrooms', 'bathrooms']]
y = df['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model trained successfully!")
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Save the model
with open('model/trained_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved to model/trained_model.pkl")