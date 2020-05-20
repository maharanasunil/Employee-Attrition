import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import random
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from matplotlib import style

train=pd.read_excel("train.xlsx")
test=pd.read_excel("test.xlsx")


train['Type']='0' #Create a flag for Train and Test Data set
test['Type']='1'
fullData = pd.concat([train,test],axis=0)
#train=0
#test=1
#print(fullData.describe())

ID_col = ['EmployeeNumber']
target_col = ["Attrition"]
cat_cols = ['Age','DailyRate','DistanceFromHome','EmployeeCount','Education','EnvironmentSatisfaction','Gender','HourlyRate','JobInvolvement','JobLevel','JobSatisfaction','MonthlyIncome','MonthlyRate','NumCompaniesWorked','OverTime','PercentSalaryHike','PerformanceRating','RelationshipSatisfaction','StandardHours','StockOptionLevel','TotalWorkingYears','TrainingTimesLastYear','WorkLifeBalance','YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager']
num_cols= list(set(list(fullData.columns))-set(cat_cols)-set(ID_col)-set(target_col))
other_col=['type'] #Test and Train Data set identifier

#to check if the data consist of any missing values
#fullData.isnull().any()

#create label encoders for categorical features
for var in cat_cols:
 number = LabelEncoder()
 fullData[var] = number.fit_transform(fullData[var].astype('str'))

#Target variable is also a categorical so convert it
#fullData["Attrition"] = number.fit_transform(fullData["Attrition"].astype('str'))


train['is_train'] = np.random.uniform(0, 1, len(train)) <= .75
Train, Validate = train[train['is_train']==True], train[train['is_train']==False]

features=list(set(list(fullData.columns))-set(ID_col)-set(target_col)-set(other_col))

x_train = Train[list(features)].values
y_train = Train["Attrition"].values
x_validate = Validate[list(features)].values
y_validate = Validate["Attrition"].values
x_test=test[list(features)].values

random.seed(100)
rf = RandomForestClassifier(n_estimators=1000)
rf.fit(x_train, y_train)

status = rf.predict_proba(x_validate)
fpr, tpr, _ = roc_curve(y_validate, status[:,1])
#print (roc_curve(y_validate, status[:,1]))
roc_auc = auc(fpr, tpr)
print (roc_auc)

final_status = rf.predict_proba(x_test)
test["Attrition1"]=final_status[:,1]
test.to_csv('output.csv',columns=['EmployeeNumber','Attrition1'])
target_col1=['Attrition1']
#style.use('ggplot')
#output=pd.read_csv("output.csv")
#plt.scatter(ID_col,target_col1)
plt.plot(['EmployeeNumber'],['Attrition1'])
plt.show()