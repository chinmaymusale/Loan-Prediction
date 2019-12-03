import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

#Import data
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")

train_original=train.copy()
test_original=test.copy()

#Understanding the data
train.columns
test.columns
train.dtypes
test.shape
train.shape

#Univariate Analysis-Target variable

train['Loan_Status'].value_counts()
train['Loan_Status'].value_counts(normalize=True)

#Univariate Analysis-Independent variable-Categorical variable

train['Gender'].value_counts(normalize=True).plot.bar(title='Gender')
train['Married'].value_counts(normalize=True).plot.bar(title='Married')
train['Self_Employed'].value_counts(normalize=True).plot.bar(title='Self_Employed')
train['Credit_History'].value_counts(normalize=True).plot.bar(title='Credit_History')
train['Property_Area'].value_counts(normalize=True).plot.bar(title='Property_Area')
plt.show()

#Univariate Analysis-Independent variable-Numerical variable

sns.distplot(train['ApplicantIncome'])
train['ApplicantIncome'].plot.box(figsize(16, 5))
plt.show()

train.boxplot(column='ApplicantIncome', by='Education')
plt.suptitle("")


train['LoanAmount'].plot.box()

#Bivariate Analysis- Categorical variable vs Target variable

Gender=pd.crosstab(train['Gender'], train['Loan_Status'])
Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True, figsize=(4,4))

Married=pd.crosstab(train['Married'], train['Loan_Status'])
Married.div(Married.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True, figsize=(4,4))

Self_Employed=pd.crosstab(train['Self_Employed'], train['Loan_Status'])
Self_Employed.div(Self_Employed.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True, figsize=(4,4))

Credit_History=pd.crosstab(train['Credit_History'], train['Loan_Status'])
Credit_History.div(Credit_History.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True, figsize=(4,4))

Property_Area=pd.crosstab(train['Property_Area'], train['Loan_Status'])
Property_Area.div(Property_Area.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True, figsize=(4,4))

#Bivariate Analysis- Categorical variable vs Target variable

train.groupby('Loan_Status')['ApplicantIncome'].mean().plot.bar()

bins=[0, 1000, 3000, 42000]
group=['Low', 'Average', 'High']
train['Coapplicant_Income_bin']=pd.cut(train['CoapplicantIncome'],bins,labels=group)
Coapplicant_Income_bin=pd.crosstab(train['Coapplicant_Income_bin'], train['Loan_Status'])
Coapplicant_Income_bin.div(Coapplicant_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True, figsize=(4,4))

train['Total_Income']=train['CoapplicantIncome'] + train['ApplicantIncome']
bins=[0, 2500, 4000, 6000, 81000]
group=['Low', 'Average', 'High', 'Very High']
train['Total_Income_bin']=pd.cut(train['Total_Income'],bins,labels=group)
Total_Income_bin=pd.crosstab(train['Total_Income_bin'], train['Loan_Status'])
Total_Income_bin.div(Total_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True, figsize=(4,4))

bins=[0, 100, 200, 700]
group=['Low', 'Average', 'High']
train['LoanAmount_bin']=pd.cut(train['LoanAmount'],bins,labels=group)
LoanAmount_bin=pd.crosstab(train['LoanAmount_bin'], train['Loan_Status'])
LoanAmount_bin.div(LoanAmount_bin.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True, figsize=(4,4))

train=train.drop(['Coapplicant_Income_bin', 'Total_Income', 'Total_Income_bin', 'LoanAmount_bin'], axis=1)
train['Dependents'].replace('3+', 3,inplace=True) 
test['Dependents'].replace('3+', 3,inplace=True) 
train['Loan_Status'].replace('N', 0,inplace=True)
train['Loan_Status'].replace('Y', 1,inplace=True)

#Missing value imputation=Categorical independent variable

train.isnull().sum()

train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)

test.isnull().sum()

test['Gender'].fillna(test['Gender'].mode()[0], inplace=True)
test['Dependents'].fillna(test['Dependents'].mode()[0], inplace=True)
test['Self_Employed'].fillna(test['Self_Employed'].mode()[0], inplace=True)
test['Credit_History'].fillna(test['Credit_History'].mode()[0], inplace=True)

#Missing value imputation=Numerical independent variable

train['Loan_Amount_Term'].value_counts()
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)

test['Loan_Amount_Term'].value_counts()
test['Loan_Amount_Term'].fillna(test['Loan_Amount_Term'].mode()[0], inplace=True)
test['LoanAmount'].fillna(test['LoanAmount'].median(), inplace=True)

#Outlier Treatment

train['LoanAmount_log']=np.log(train['LoanAmount'])
train['LoanAmount_log'].hist(bins=20)
test['LoanAmount_log']=np.log(test['LoanAmount'])

#Model building

train=train.drop('Loan_ID', axis=1)
test=test.drop('Loan_ID', axis=1)

X = train.drop(['Loan_Status'], axis=1)
y = train['Loan_Status']

X=pd.get_dummies(X)
train=pd.get_dummies(train)
test=pd.get_dummies(test)

from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(X, y, test_size=0.3)

from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score

model=LogisticRegression()
model.fit(x_train, y_train)
LogisticRegression(multi_class='ovr', n_jobs=1, random_state=1, solver='liblinear', tol=0.0001)

pred_cv = model.predict(x_cv)
accuracy_score(y_cv, pred_cv)

pred_test=model.predict(test)
submission=pd.read_csv("sample_submission.csv")

submission['Loan_Status']= pred_test
submission['Loan_ID']= test_original['Loan_ID']

submission['Loan_Status'].replace(0, 'N', inplace=True)
submission['Loan_Status'].replace(1, 'Y', inplace=True)


































