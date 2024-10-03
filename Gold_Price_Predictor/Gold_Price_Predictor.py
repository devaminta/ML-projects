import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# Loading the CSV data into a Pandas DataFrame
gold_data = pd.read_csv('gld_price_data.csv')

# Print first 5 rows in the dataframe
print(gold_data.head())

# Print last 5 rows of the dataframe
print(gold_data.tail())

# Number of rows and columns
print(gold_data.shape)

# Getting some basic information about the data
print(gold_data.info())

# Checking the number of missing values
print(gold_data.isnull().sum())

# Handling missing values (if any)
gold_data.dropna(inplace=True)

# Getting the statistical measures of the data
print(gold_data.describe())

# Exclude the 'Date' column for correlation calculation
correlation = gold_data.drop(columns=['Date']).corr()


# Constructing a heatmap to understand the correlation
plt.figure(figsize=(8, 8))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size': 8}, cmap='Blues')
plt.title('Correlation Heatmap')
plt.show()

# Correlation values of GLD
print(correlation['GLD'])

# Checking the distribution of the GLD Price
sns.histplot(gold_data['GLD'], kde=True, color='green')
plt.title('Distribution of GLD Prices')
plt.xlabel('GLD Price')
plt.ylabel('Frequency')
plt.show()

# Preparing the data for training
X = gold_data.drop(['Date', 'GLD'], axis=1)
Y = gold_data['GLD']

# Splitting the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Creating the regressor model
regressor = RandomForestRegressor(n_estimators=100)

# Training the model
regressor.fit(X_train, Y_train)

# Prediction on Test Data
test_data_prediction = regressor.predict(X_test)

# R squared error
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared error: ", error_score)

# Calculate additional metrics
mae = metrics.mean_absolute_error(Y_test, test_data_prediction)
mse = metrics.mean_squared_error(Y_test, test_data_prediction)
print("Mean Absolute Error: ", mae)
print("Mean Squared Error: ", mse)

# Plotting Actual vs Predicted values
plt.plot(Y_test.values, color='blue', label='Actual Value')
plt.plot(test_data_prediction, color='green', label='Predicted Value')
plt.title('Actual Price vs Predicted Price')
plt.xlabel('Number of Values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()