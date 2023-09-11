import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')
import seaborn as sns
from sklearn import preprocessing
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

data = pd.read_csv("Dataset/healthcare-dataset-stroke-data.csv")
# Success
# print("Stroke dataset has {} data points with {} variables each.".format(*data.shape))
# print(data)

print(data.hypertension.value_counts())

sns.set_theme(style="darkgrid")
ax = sns.countplot(data=data, x="hypertension")

# For Dataset Section
# plt.show()

# *** Data Preprocessing ***
mean_bmi = data['bmi'].mean()
data['bmi'].fillna(value=mean_bmi, inplace=True)
# scaler = StandardScaler()

# Define the list of numeric columns to scale
numeric_cols = ['age', 'avg_glucose_level', 'bmi']

# Scale the numeric columns in the 'data' dataframe using the 'fit_transform' method of the StandardScaler object
# data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# Label Encoding
label_encoder = preprocessing.LabelEncoder()

data['Sex'] = label_encoder.fit_transform(data['gender'])
data['Married'] = label_encoder.fit_transform(data['ever_married'])
data['Employment'] = label_encoder.fit_transform(data['work_type'])
data['Residency'] = label_encoder.fit_transform(data['Residence_type'])
data['Smoker'] = label_encoder.fit_transform(data['smoking_status'])
data = data.drop(['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'], axis=1)
data.drop('id', axis=1, inplace=True)

# Apply Corelation Matrix After preprocessing

plt.figure(figsize=(12, 8))
ax = sns.heatmap(data.corr(), annot=True)
plt.show()


# Define a function that takes a dataset and a threshold value as inputs and returns the set of all names of correlated columns
def correlation(dataset, threshold):
    # Initialize an empty set to store the names of correlated columns
    col_corr = set()

    # Calculate the correlation matrix for the dataset using the 'corr' method
    corr_matrix = dataset.corr()

    # Iterate over the columns of the correlation matrix
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            # Check if the absolute value of the correlation coefficient is greater than the threshold
            if abs(corr_matrix.iloc[i, j]) > threshold:
                # If it is, add the name of the column to the set of correlated columns
                colname = corr_matrix.columns[i]
                col_corr.add(colname)

    # Return the set of correlated column names
    return col_corr


corr_features = correlation(data, 0.35)

X_corr = data.drop(corr_features, axis=1)

# splitting data

# drop the 'Stroke' column from the DataFrame to create the feature matrix
X = data.drop('stroke', axis=1)
# create the target vector
y = data['stroke']

# split the data into training and testing subsets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=4)
x_train, y_train = sm.fit_resample(x_train, y_train.ravel())


def Model(model):
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    model_train_score = model.score(x_train, y_train)
    model_test_score = model.score(x_test, y_test)
    prediction = model.predict(x_test)

    cm = confusion_matrix(y_test, prediction)
    precision = cm[0][0] / (cm[0][0] + cm[1][0])
    recall = cm[0][0] / (cm[0][0] + cm[0][1])
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('F1 score: ', 2 * (precision * recall) / (precision + recall))
    print('Testing Score \n', score)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()
    y_score = model.predict_proba(x_test)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)
    fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC={metrics.auc(fpr, tpr):.4f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=700, height=500
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    plt.show()


# Decision Tree Classification
d_classif = DecisionTreeClassifier()
print("Decision Tree")
Model(d_classif)

# Logistic Regression
lg_reg = LogisticRegression()
print("Logistic Regression")
Model(lg_reg)
