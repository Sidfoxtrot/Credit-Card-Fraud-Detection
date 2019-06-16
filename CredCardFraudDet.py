from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score,auc
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import metrics
data = pd.read_csv('creditcard.csv')
#data.head()
data['Scaled_Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
data['Scaled_Time'] = StandardScaler().fit_transform(data['Time'].values.reshape(-1,1))
data.drop(['Time', 'Amount'], axis=1, inplace=True)
#data.head()
X = data.iloc[:,data.columns != "Class"]
Y = data.iloc[:,data.columns == "Class"]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=50)
clf= DecisionTreeClassifier()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print(classification_report(y_test, pred))
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
print (roc_auc)
