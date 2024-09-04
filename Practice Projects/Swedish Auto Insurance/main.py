import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = '../../Practice Projects/Datasets/Swedish Auto Insurance Dataset.xls'
data = pd.read_excel(file_path, engine='xlrd')

# Split the dataset into input (X) and output (Y) variables
X = data[['X']]
Y = data['Y']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train a regression model on the training data
model = LinearRegression()
model.fit(X_train, Y_train)

# Make predictions on the testing data
Y_pred = model.predict(X_test)

# Calculate the RMSE of the predictions
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))

# Print the RMSE
print(f'RMSE: {rmse:.2f} thousand Kronor')

# Plot the original data points
plt.scatter(X, Y, color='blue', label='Original data')

# Plot the regression line based on the model
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regression line')

# Plot the predictions vs actual values
plt.scatter(X_test, Y_test, color='green', label='Actual values')
plt.scatter(X_test, Y_pred, color='orange', label='Predicted values')

# Add labels and legend
plt.xlabel('Number of claims')
plt.ylabel('Total payment for all claims (in thousands of Kronor)')
plt.title('Swedish Auto Insurance Data')
plt.legend()

# Show the plot
plt.show()
