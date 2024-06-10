import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

COLUMNS_TO_DROP = ['Name', 'PassengerId', 'Cabin', 'Ticket']

train_dataset = pd.read_csv('datasets/train.csv')
train_dataset = train_dataset.drop(COLUMNS_TO_DROP, axis = 1)
x_train = train_dataset.drop(['Survived'], axis = 1).values
y_train = train_dataset['Survived'].values

test_dataset = pd.read_csv('datasets/test.csv')
test_dataset = test_dataset.drop(COLUMNS_TO_DROP, axis = 1)
x_test = test_dataset.values


le = LabelEncoder()
x_train[:, 1] = le.fit_transform(x_train[:, 1])
x_test[:, 1] = le.fit_transform(x_test[:, 1])
x_train[:, -1] = le.fit_transform(x_train[:, -1])
x_test[:, -1] = le.fit_transform(x_test[:, -1])


si = SimpleImputer(missing_values = np.nan, strategy = 'mean')
x_test = si.fit_transform(x_test)
x_train = si.fit_transform(x_train)


clasification = LogisticRegression(C = 10, random_state = 0, max_iter = 1000)
clasification.fit(x_train, y_train)
y_pred = clasification.predict(x_test)
y_test = pd.read_csv('datasets/gender_submission.csv').drop('PassengerId', axis = 1).values

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))