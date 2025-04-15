import pandas as pd
import warnings
import numpy as np
from sklearn.preprocessing import RobustScaler,OneHotEncoder
warnings.filterwarnings('ignore')

# read data
data = pd.read_csv('training_data.csv')
data = data.drop(['Unnamed: 0'], axis=1)

categorical_data = data.select_dtypes(include=['object']).columns.tolist()
numerical_data = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
y_data = data['y']

# one-hot encoding
encoder = OneHotEncoder(sparse_output=False)
one_hot_encoded = pd.DataFrame(encoder.fit_transform(data[categorical_data]), columns=encoder.get_feature_names_out(categorical_data))
data_encoded = pd.concat([data[numerical_data], one_hot_encoded], axis=1)

# data_encoded['balance_log'] = np.log(data_encoded['balance'] + 8019)
# data_encoded.drop(['balance'], axis=1, inplace=True)

# normalization
scaler = RobustScaler()
scaled_data = scaler.fit_transform(data_encoded)
data_encoded_normalised = pd.DataFrame(scaled_data, columns=data_encoded.columns)


data_encoded_normalised['y'] = y_data
data_encoded_normalised.drop(['y_no'], axis=1, inplace=True)
data_encoded_normalised.drop(['y_yes'], axis=1, inplace=True)
data_encoded_normalised.drop(['job_unknown'], axis=1, inplace=True)
data_encoded_normalised.drop(['marital_divorced'], axis=1, inplace=True)
data_encoded_normalised.drop(['education_unknown'], axis=1, inplace=True)
data_encoded_normalised.drop(['default_yes'], axis=1, inplace=True)
data_encoded_normalised.drop(['housing_yes'], axis=1, inplace=True)
data_encoded_normalised.drop(['loan_yes'], axis=1, inplace=True)
data_encoded_normalised.drop(['contact_unknown'], axis=1, inplace=True)
data_encoded_normalised.drop(['poutcome_unknown'], axis=1, inplace=True)

data_encoded_normalised.to_csv('training_data_preprocessed.csv', index=False)


