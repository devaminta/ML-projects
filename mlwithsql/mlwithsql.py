import pandas as pd
import pymysql
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Connect to the database
connection = pymysql.connect(
    host='localhost',
    user='username',
    password='password',
    database='database_name'
)

# Read data into a Pandas DataFrame
df = pd.read_sql('SELECT * FROM table_name', con=connection)

# Prepare the data
X = df[['feature_1', 'feature_2']]
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Close the connection
connection.close()