import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import nltk

nltk.download("stopwords")

# Loading the dataset into the pandas DataFrame
news_dataset = pd.read_csv("train.csv")

# Print the first 5 rows of the dataframe
print(news_dataset.head())

# Counting the number of missing values in the dataset
print(news_dataset.isnull().sum())

# Replacing the null values with empty string
news_dataset = news_dataset.fillna('')

# Merging the author name and news title
news_dataset["content"] = news_dataset["author"] + " " + news_dataset["title"]
print(news_dataset["content"])

# Separating the data and label
X = news_dataset["content"].values
y = news_dataset["label"].values  # Corrected this line

# Stemming function
port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub("[^a-zA-Z]", " ", content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()  # Corrected this line
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words("english")]
    stemmed_content = " ".join(stemmed_content)
    return stemmed_content

news_dataset["content"] = news_dataset["content"].apply(stemming)
print(news_dataset["content"])

# Vectorizing the function (converting the textual data to numerical data)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Splitting the dataset into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Fitting the logistic regression to the training data
classifier = LogisticRegression()
classifier.fit(X_train, y_train)  # Corrected this line

# Predicting the test data
x_train_pred = classifier.predict(X_train)
x_test_pred = classifier.predict(X_test)

# Let’s see the accuracies and the confusion matrix
cm = confusion_matrix(y_test, x_test_pred)
print(cm)

accuracy_of_training_data = accuracy_score(y_train, x_train_pred)
accuracy_of_test_data = accuracy_score(y_test, x_test_pred)

print("Accuracy score of the training data:", accuracy_of_training_data)
print("Accuracy score of the test data:", accuracy_of_test_data)

# Visualization part can be skipped or modified based on dimensionality