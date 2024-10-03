# Diabetes Prediction 
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# the datset
dataset=pd.read_csv("Diabetes.csv")

X=dataset.drop(columns="outcome",axis=1)
y=dataset["outcome"]
# X=dataset.iloc[:,-1].values
# y=dataset.iloc[:,1].values


# dataset["outcome"].value_counts()

# dataset.groupby("Outcome").mean()
scaler=StandardScaler()
standardized_data=scaler.fit_transform()

# split training and test set data
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=2,test_size=0.2,stratify=y)


classifier=SVC(kernel="linear")
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

# checking the accuracy score
data_accuracy_score=accuracy_score(y_test,y_pred)



# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_



