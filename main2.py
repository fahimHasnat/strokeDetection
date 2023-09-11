import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

df = pd.read_csv("Dataset/healthcare-dataset-stroke-data.csv")

# Data Preprocessing
df.drop(['id'], axis=1, inplace=True)
df.drop_duplicates(inplace=True)

df.dropna(inplace=True)

df.isnull().sum().sum()
num_cols = ['age', 'bmi', 'avg_glucose_level']

plt.figure(figsize=(15, 5))
for i in range(3):
    plt.subplot(1, 3, i + 1)

    sns.boxplot(x=df[num_cols[i]], color='#6DA59D')
    plt.title(num_cols[i])
plt.show()


def detect_outliers(data, column):
    q1 = df[column].quantile(.25)
    q3 = df[column].quantile(.75)
    IQR = q3 - q1

    lower_bound = q1 - (1.5 * IQR)
    upper_bound = q3 + (1.5 * IQR)

    ls = df.index[(df[column] < lower_bound) | (df[column] > upper_bound)]

    return ls


index_list = []

for column in num_cols:
    index_list.extend(detect_outliers(df, column))

# remove duplicated indices in the index_list and sort it
index_list = sorted(set(index_list))


