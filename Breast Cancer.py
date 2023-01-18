# %%
# Importing required libraries
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')

# %%
# loading the dataset
data = pd.read_csv("breastCancer.csv")
data.head()

# %%
# checking the shape of the data
data.shape

# %%
# check the data type
data.dtypes

# %% [markdown]
# All the columns are integer including "bare_nucleoli" but when we checked the data type it actually telling the "bare_nucleoli" is object. The reason is that something inside the column that not an integer value and need to take care of that value.

# %%
bare_ = pd.DataFrame(data['bare_nucleoli'].str.isdigit())

# %%
data[bare_['bare_nucleoli'] == False]

# %%
# replace the '?' with NaN
data = data.replace('?', np.nan)

# %%
# Check the median value
med = data['bare_nucleoli'].median()
med

# %%
# replace the question mark with median value
data['bare_nucleoli'] = data['bare_nucleoli'].fillna(med)

# %%
data.dtypes

# %% [markdown]
# We replaced '?' mark with median value(1) and still the data type is showing object type, handle that.

# %%
data['bare_nucleoli'] = data['bare_nucleoli'].astype('int64')

# %%
data.info()

# %%
# check duplicate values
data.duplicated().values.any()

# %% [markdown]
# here we have some duplicate values in this dataset we have to drop that rows.

# %%
# dropping duplicate rows
data = data.drop_duplicates()

# %%
data.duplicated().any()

# %%
data.head()

# %% [markdown]
# we don't need the 'id' column and we have to drop that column.

# %%
data.drop('id', axis=1, inplace=True)

# %%
data.head()

# %%
# checking the 5 number summary.
data.describe().T

# %%
# checking null values 
sns.heatmap(data.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.show()

# %% [markdown]
# its good we have no null values.

# %%
# checking unique values in target column('class')
data['class'].value_counts()

# %%
# countplot
f, ax = plt.subplots(1, 2, figsize=(15, 7))
ax[0] = data['class'].value_counts().plot.pie(explode=[0,0], autopct='%1.1f%%', ax=ax[0], shadow=True)
ax[0].set_title("Class")

ax[1] = sns.countplot(x='class', data=data, palette='Set1')
ax[1].set_title('Frequency Distribution of class')
plt.show()

# %%
# histplot
plt.figure(figsize=(15, 20))
col_list = data.columns

for i in range(len(data.columns)):
    plt.subplot(5, 2, i+1)
    plt.hist(data[col_list[i]], edgecolor='k', color='orange', bins=20)
    plt.title(col_list[i], color='black', fontsize=15)
    plt.tight_layout()
plt.show()

# %%
# boxplot 
plt.figure(figsize=(10,10))
sns.boxplot(data=data, orient='h');

# %%
# check the correlation
data.corr()

# %% [markdown]
# columns are positively correlated some are very high and some are low.

# %%
# heatmap
plt.figure(figsize=(20,10))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f', square=True, linecolor='k', linewidths=0.5, vmax=1)
plt.title('Correlation Heatmap');

# %%
# pairplot
sns.pairplot(data=data, diag_kind='kde')
plt.title('Distribution of variables')
plt.show()

# %% [markdown]
# ### **Model Building**

# %%
data.head()

# %%
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# %% [markdown]
# ### **K-Nearest Neighbor Classification**

# %%
from sklearn.neighbors import KNeighborsClassifier 

KNN = KNeighborsClassifier(n_neighbors=5, weights='distance')
KNN.fit(X_train, y_train)

# %%
KNN_pred = KNN.predict(X_test)
KNN_pred

# %%
# check the score
print('KNN predicted score: {0:.2f}%'.format(KNN.score(X_test, y_test)*100))

# %%
# choosing best k 

error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10, 6))
plt.plot(range(1,40), error_rate, color='blue', linestyle='dashed',
         marker='o', markerfacecolor='red', markersize=10)
plt.title('Error Rate Vs. K value')
plt.xlabel('K')
plt.ylabel('Error rate')
print('Minimum Error:', min(error_rate), "at K=", error_rate.index(min(error_rate)));

# %%
acc = []
# Will take some time
from sklearn import metrics
for i in range(1,40):
    neigh = KNeighborsClassifier(n_neighbors = i).fit(X_train,y_train)
    yhat = neigh.predict(X_test)
    acc.append(metrics.accuracy_score(y_test, yhat))
    
plt.figure(figsize=(10,6))
plt.plot(range(1,40),acc,color = 'blue',linestyle='dashed', 
         marker='o',markerfacecolor='red', markersize=10)
plt.title('accuracy vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy')
print("Maximum accuracy:-",max(acc),"at K =",acc.index(max(acc)))

# %% [markdown]
# - k=5 is showing the better accuracy.

# %%
# classification report
from sklearn import metrics

print(metrics.classification_report(y_test, KNN_pred))

# %%
# confusion matrix
cm = metrics.confusion_matrix(y_test, KNN_pred, labels=[2, 4])
df_cm = pd.DataFrame(cm, index=[i for i in ["2", "4"]],
                     columns=[i for i in ['Predicted 2', 'Predicted 4']])
plt.figure(figsize=(7,5))
sns.heatmap(df_cm, annot=True,fmt='.2f', square=True)
plt.title('Confusion Matrix-KNN');

# %% [markdown]
# ### **Support Vector Machine(SVC)**

# %%
from sklearn import svm

svc = svm.SVC(C=3, random_state=42,gamma=0.005, kernel='linear')
svc.fit(X_train, y_train)

# %%
# predict the test set result
svc_pred = svc.predict(X_test)
svc_pred

# %%
print('Predicted score of SVC: {0:.2f}%'.format(metrics.accuracy_score(y_test, svc_pred)*100))

# %% [markdown]
# The KNN accuracy score is better than SVC.

# %%
# classification report of svc
print(metrics.classification_report(y_test, svc_pred))

# %%
# confusion matrix of svc
cm = metrics.confusion_matrix(y_test, svc_pred, labels=[2,4])
df_cm = pd.DataFrame(cm, index=[i for i in ["2", "4"]],
                     columns=[i for i in ['Predicted 2', 'Predicted 4']])
plt.figure(figsize=(7,5))
sns.heatmap(df_cm, annot=True, square=True, fmt='.2f') 
plt.title('Confusion Matrix-SVC');

# %%
# compare these predictions
knn_predictions = pd.DataFrame(KNN_pred)
svc_predictions = pd.DataFrame(svc_pred)

# %%
pred_df = pd.concat([knn_predictions, svc_predictions], axis=1)
pred_df.columns = [['knn_predictions', 'svc_predictions']]

# %%
pred_df.head()


