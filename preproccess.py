import pandas as pd
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
warnings.filterwarnings('ignore')

data = pd.read_csv('training_data.csv')
data = data.drop(['Unnamed: 0'], axis=1)

categorical_data = data.select_dtypes(include=['object']).columns.tolist()
numerical_data = data.select_dtypes(include=['int64', 'float64']).columns.tolist()


def outliers(column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] < lower_bound) | (data[column] > upper_bound)]


# change binary category data to numerical
data['y'] = data['y'].map({'yes': 1, 'no': 0})
data['default'] = data['default'].map({'yes': 1, 'no': 0})
data['housing'] = data['housing'].map({'yes': 1, 'no': 0})
data['loan'] = data['loan'].map({'yes': 1, 'no': 0})

# use one-hot to make the data machine readable
jobs = ['management', 'technician', 'entrepreneur', 'blue-collar', 'unknown', 'retired',
        'admin.', 'services', 'self-employed', 'unemployed', 'housemaid', 'student']
married = ['single', 'divorced']
education = ['tertiary', 'secondary', 'unknown', 'primary']


def onehot(x, options):
    template = [0 for _ in range(len(options))]
    template[options.index(x)] = 1
    return template


# inputs = [0.5, 0.1, 0.3, 1, 0]
# print(f"Data Before Adding Job one-hot: {inputs}")
# job = 'technician'
# inputs.extend(onehot(job, jobs))
# print(f"Data After Adding Job one-hot: {inputs}")
