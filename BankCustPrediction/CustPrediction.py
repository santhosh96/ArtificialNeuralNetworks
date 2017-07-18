# Artificial Neural Network for prediction wether a customer would leave the Bank. The data set is given in the csv file


# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Creating ANN

# Import Keras library
import tensorflow
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and first hidden layer
classifier.add(Dense(output_dim = 6, kernel_initializer = 'uniform', activation= 'relu', input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, kernel_initializer = 'uniform', activation= 'relu'))

# Adding output layer
classifier.add(Dense(output_dim = 1, kernel_initializer = 'uniform', activation= 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])

# Fitting the ANN to training set
classifier.fit(X_train, y_train, batch_size= 100, epochs= 100)

# Predicting the test set
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Confusion metrics
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

# % Accuracy 
add = cm[0][0] + cm[1][1]
acc = add / (add + cm[0][1] + cm[1][0])

"""Prediction of the customer with following details
Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $60000
Number of Products: 2
Does this customer have a credit card ? Yes
Is this customer an Active Member: Yes
Estimated Salary: $50000"""

new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)

#False the customer won't leave the bank