import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier, BaggingClassifier, StackingClassifier

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

for column in num_cols:
    index_list.extend(detect_outliers(df, column))

# remove duplicated indices in the index_list and sort it
index_list = sorted(set(index_list))

before_remove = df.shape

df = df.drop(index_list)
after_remove = df.shape

print(f'''Shape of data before removing outliers : {before_remove}
Shape of data after remove : {after_remove}''')

df_0 = df[df.iloc[:, -1] == 0]
df_1 = df[df.iloc[:, -1] == 1]

df['stroke'].value_counts()

from sklearn.utils import resample

df_1 = resample(df_1, replace=True, n_samples=df_0.shape[0], random_state=123)
df = np.concatenate((df_0, df_1))

# create the balanced dataframe
df = pd.DataFrame(df)
df.columns = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type',
              'avg_glucose_level', 'bmi', 'smoking_status', 'stroke']

# visualize balanced data
stroke = dict(df['stroke'].value_counts())
fig = px.pie(names=['False', 'True'], values=stroke.values(), title='Stroke Occurance',
             color_discrete_sequence=px.colors.sequential.Aggrnyl)
fig.update_traces(textposition='inside', textinfo='percent+label')

df = pd.get_dummies(data=df, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'],
                    drop_first=True)

# Splitting
x = df.drop('stroke', axis=1)
y = pd.to_numeric(df['stroke'])

scaler = StandardScaler()

x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.20)

# Decision Tree
print("Decision Tree")
tree_model = DecisionTreeClassifier(criterion='entropy')
tree_model.fit(x_train, y_train)

y_pred = tree_model.predict(x_test)

accuracy_score(y_test, y_pred)

# KNN
print("Knn")
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
accuracy_score(y_test, y_pred)

# Naive Bayes
NB_model = GaussianNB()
NB_model.fit(x_train, y_train)
y_pred = NB_model.predict(x_test)
accuracy_score(y_test, y_pred)

# SVC
svm = SVC()
svm.fit(x_train, y_train)

y_pred = svm.predict(x_test)
accuracy_score(y_test, y_pred)

# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(x_train, y_train)

y_pred = lr_model.predict(x_test)
accuracy_score(y_test, y_pred)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=150,criterion='entropy',random_state = 123)
rf_model.fit(x_train,y_train)

y_pred = rf_model.predict(x_test)
accuracy_score(y_test,y_pred)

# Voting Technique
svm = SVC()
LR = LogisticRegression()
tree = DecisionTreeClassifier()
knn = KNeighborsClassifier(n_neighbors=3)

models = [('SVM',svm),('Decision Tree',tree),('Logistic Regerssion',LR) , ('KNN',knn)]

voting_model = VotingClassifier(
    estimators= models
)

y_pred = voting_model.predict(x_test)
accuracy_score(y_test,y_pred)

# Bagging Model
bagging = BaggingClassifier(
    estimator = knn,
    n_estimators = 10
)

bagging.fit(x_train , y_train)

y_pred = bagging.predict(x_test)
accuracy_score(y_test,y_pred)

# Stacking Technique
base_models = [('SVM',SVC()),('Decision Tree',DecisionTreeClassifier()),('Logistic Regerssion',LogisticRegression()) , ('KNN',KNeighborsClassifier(n_neighbors=3))]
stacking = StackingClassifier(
    estimators = base_models ,
    final_estimator = LogisticRegression(),
    cv = 5
)

stacking.fit(x_train , y_train)

y_pred = stacking.predict(x_test)
accuracy_score(y_test,y_pred)
