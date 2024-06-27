import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

data = pd.read_csv('training_data.csv')
data = data.drop(['Unnamed: 0'], axis=1)

print(data.isnull().sum())