import pandas as pd
import warnings
from sklearn.preprocessing import Normalizer,OneHotEncoder
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

# read data
data = pd.read_csv('training_data.csv')
data = data.drop(['Unnamed: 0'], axis=1)

categorical_data = data.select_dtypes(include=['object']).columns.tolist()
numerical_data = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

# one-hot encoding
encoder = OneHotEncoder(sparse_output=False)
one_hot_encoded = pd.DataFrame(encoder.fit_transform(data[categorical_data]), columns=encoder.get_feature_names_out(categorical_data))
data_encoded = pd.concat([data[numerical_data], one_hot_encoded], axis=1)

# normalization
scaler = Normalizer()
scaled_data = scaler.fit_transform(data_encoded)
data_encoded_normalised = pd.DataFrame(scaled_data, columns=data_encoded.columns)

data_encoded_normalised.to_csv('training_data_preprocessed.csv', index=False)


