import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import confusion_matrix, classification_report 
from sklearn.model_selection import train_test_split 
import statsmodels.api as sm

data = pd.read_csv('training_data_preprocessed.csv')

def outliers(column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] < lower_bound) | (data[column] > upper_bound)].shape[0]

X_train, X_test, y_train, y_test = train_test_split(data.drop(['y'], axis=1), data['y'], test_size=0.2, random_state=0)

sm = SMOTE()
X_train_res,y_train_res = sm.fit_resample(X_train, y_train)

model = RandomForestClassifier()
model.fit(X_train_res, y_train_res)
predictions = model.predict(X_test)

print(outliers('balance'))
print(classification_report(y_test, predictions))