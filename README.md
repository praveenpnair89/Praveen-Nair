# Praveen-Nair
Predicting the Survival of Titanic Passengers
%matplotlib inline
import math
import random
import re, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
pip install pandas-summary
Requirement already satisfied: pandas-summary in c:\users\asus\anaconda3\lib\site-packages (0.0.7)
Requirement already satisfied: pandas in c:\users\asus\anaconda3\lib\site-packages (from pandas-summary) (0.24.2)
Requirement already satisfied: numpy in c:\users\asus\anaconda3\lib\site-packages (from pandas-summary) (1.16.4)
Requirement already satisfied: pytz>=2011k in c:\users\asus\anaconda3\lib\site-packages (from pandas->pandas-summary) (2019.1)
Requirement already satisfied: python-dateutil>=2.5.0 in c:\users\asus\anaconda3\lib\site-packages (from pandas->pandas-summary) (2.8.0)
Requirement already satisfied: six>=1.5 in c:\users\asus\anaconda3\lib\site-packages (from python-dateutil>=2.5.0->pandas->pandas-summary) (1.12.0)
Note: you may need to restart the kernel to use updated packages.
from pandas_summary import DataFrameSummary
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
drt = pd.read_csv("data1/rmstitanic.csv")
drt.head(10)
PassengerId	Survived	Pclass	Name	Sex	Age	SibSp	Parch	Ticket	Fare	Cabin	Embarked
0	1	0	3	Braund, Mr. Owen Harris	male	22.0	1	0	A/5 21171	7.2500	NaN	S
1	2	1	1	Cumings, Mrs. John Bradley (Florence Briggs Th...	female	38.0	1	0	PC 17599	71.2833	C85	C
2	3	1	3	Heikkinen, Miss. Laina	female	26.0	0	0	STON/O2. 3101282	7.9250	NaN	S
3	4	1	1	Futrelle, Mrs. Jacques Heath (Lily May Peel)	female	35.0	1	0	113803	53.1000	C123	S
4	5	0	3	Allen, Mr. William Henry	male	35.0	0	0	373450	8.0500	NaN	S
5	6	0	3	Moran, Mr. James	male	NaN	0	0	330877	8.4583	NaN	Q
6	7	0	1	McCarthy, Mr. Timothy J	male	54.0	0	0	17463	51.8625	E46	S
7	8	0	3	Palsson, Master. Gosta Leonard	male	2.0	3	1	349909	21.0750	NaN	S
8	9	1	3	Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)	female	27.0	0	2	347742	11.1333	NaN	S
9	10	1	2	Nasser, Mrs. Nicholas (Adele Achem)	female	14.0	1	0	237736	30.0708	NaN	C
print(pd.isnull(drt).sum()) #cabin can be removed as 77% of data are missing
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
dtype: int64
#dropping ticket as not useful
drt = drt.drop(['Ticket'], axis = 1)
drt.columns.values
array(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Fare', 'Cabin', 'Embarked'], dtype=object)
## CORRELATION OF AGE AND GENDER

#draw a bar plot of survival by sex
sns.barplot(x="Sex", y="Survived", data=drt)

#print percentages of females vs. males that survive
print("Percentage of females who survived:", drt["Survived"][drt["Sex"] == 'female'].value_counts(normalize = True)[1]*100)

print("Percentage of males who survived:", drt["Survived"][drt["Sex"] == 'male'].value_counts(normalize = True)[1]*100)
Percentage of females who survived: 74.20382165605095
Percentage of males who survived: 18.890814558058924

#draw a bar plot of survival by Pclass
sns.barplot(x="Pclass", y="Survived", data=drt)

#print percentage of people by Pclass that survived
print("Percentage of Pclass = 1 who survived:", drt["Survived"][drt["Pclass"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of Pclass = 2 who survived:", drt["Survived"][drt["Pclass"] == 2].value_counts(normalize = True)[1]*100)

print("Percentage of Pclass = 3 who survived:", drt["Survived"][drt["Pclass"] == 3].value_counts(normalize = True)[1]*100)
Percentage of Pclass = 1 who survived: 62.96296296296296
Percentage of Pclass = 2 who survived: 47.28260869565217
Percentage of Pclass = 3 who survived: 24.236252545824847

#draw a bar plot for SibSp vs. survival
sns.barplot(x="SibSp", y="Survived", data=drt)

print("Percentage of SibSp = 0 who survived:", drt["Survived"][drt["SibSp"] == 0].value_counts(normalize = True)[1]*100)

print("Percentage of SibSp = 1 who survived:", drt["Survived"][drt["SibSp"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of SibSp = 2 who survived:", drt["Survived"][drt["SibSp"] == 2].value_counts(normalize = True)[1]*100)
Percentage of SibSp = 0 who survived: 34.53947368421053
Percentage of SibSp = 1 who survived: 53.588516746411486
Percentage of SibSp = 2 who survived: 46.42857142857143

#draw a bar plot for Parch vs. survival
sns.barplot(x="Parch", y="Survived", data=drt)
plt.show()

#sort the ages into logical categories
drt["Age"] = drt["Age"].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, np.inf]
labels = ['Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
drt['AgeGroup'] = pd.cut(drt["Age"], bins, labels = labels)

#draw a bar plot of Age vs. survival
sns.barplot(x="AgeGroup", y="Survived", data=drt)
plt.show()

drt["CabinBool"] = (drt["Cabin"].notnull().astype('int'))

#calculate percentages of CabinBool vs. survived
print("Percentage of CabinBool = 1 who survived:", drt["Survived"][drt["CabinBool"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of CabinBool = 0 who survived:", drt["Survived"][drt["CabinBool"] == 0].value_counts(normalize = True)[1]*100)
#draw a bar plot of CabinBool vs. survival
sns.barplot(x="CabinBool", y="Survived", data=drt)
plt.show()
Percentage of CabinBool = 1 who survived: 66.66666666666666
Percentage of CabinBool = 0 who survived: 29.985443959243085

DATA CLEANING
drt.describe(include = 'all')
PassengerId	Survived	Pclass	Name	Sex	Age	SibSp	Parch	Fare	Cabin	Embarked	AgeGroup	CabinBool
count	891.000000	891.000000	891.000000	891	891	891.000000	891.000000	891.000000	891.000000	204	889	891	891.000000
unique	NaN	NaN	NaN	891	2	NaN	NaN	NaN	NaN	147	3	7	NaN
top	NaN	NaN	NaN	Beavan, Mr. William Thomas	male	NaN	NaN	NaN	NaN	B96 B98	S	Adult	NaN
freq	NaN	NaN	NaN	1	577	NaN	NaN	NaN	NaN	4	644	220	NaN
mean	446.000000	0.383838	2.308642	NaN	NaN	23.712121	0.523008	0.381594	32.204208	NaN	NaN	NaN	0.228956
std	257.353842	0.486592	0.836071	NaN	NaN	17.735270	1.102743	0.806057	49.693429	NaN	NaN	NaN	0.420397
min	1.000000	0.000000	1.000000	NaN	NaN	-0.500000	0.000000	0.000000	0.000000	NaN	NaN	NaN	0.000000
25%	223.500000	0.000000	2.000000	NaN	NaN	6.000000	0.000000	0.000000	7.910400	NaN	NaN	NaN	0.000000
50%	446.000000	0.000000	3.000000	NaN	NaN	24.000000	0.000000	0.000000	14.454200	NaN	NaN	NaN	0.000000
75%	668.500000	1.000000	3.000000	NaN	NaN	35.000000	1.000000	0.000000	31.000000	NaN	NaN	NaN	0.000000
max	891.000000	1.000000	3.000000	NaN	NaN	80.000000	8.000000	6.000000	512.329200	NaN	NaN	NaN	1.000000
#Dropping cabin
drt = drt.drop(['Cabin'], axis = 1)
# fill the missing values in the Embarked feature
print("Number of people embarking in Southampton (S):")
southampton = drt[drt["Embarked"] == "S"].shape[0]
print(southampton)

print("Number of people embarking in Cherbourg (C):")
cherbourg = drt[drt["Embarked"] == "C"].shape[0]
print(cherbourg)

print("Number of people embarking in Queenstown (Q):")
queenstown = drt[drt["Embarked"] == "Q"].shape[0]
print(queenstown)
Number of people embarking in Southampton (S):
644
Number of people embarking in Cherbourg (C):
168
Number of people embarking in Queenstown (Q):
77
# as most passengers embarked in southampton, replacing the missing values in the Embarked feature with S
drt = drt.fillna({"Embarked": "S"})
a = [drt]
# Cleaning AGE feature.  need to check title and gender before filling the missing values


#extract a title for the Name in datasets
for dataset in a:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(drt['Title'], drt['Sex'])
Sex	female	male
Title		
Capt	0	1
Col	0	2
Countess	1	0
Don	0	1
Dr	1	6
Jonkheer	0	1
Lady	1	0
Major	0	2
Master	0	40
Miss	182	0
Mlle	2	0
Mme	1	0
Mr	0	517
Mrs	125	0
Ms	1	0
Rev	0	6
Sir	0	1
#replace various titles with more common names
for dataset in a:
    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col',
    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'General')
    
    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

drt[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
Title	Survived
0	General	0.250000
1	Master	0.575000
2	Miss	0.702703
3	Mr	0.156673
4	Mrs	0.793651
5	Royal	1.000000
#map each of the title to a numerical value
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "General": 6}
for dataset in a:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

drt.head(50)
PassengerId	Survived	Pclass	Name	Sex	Age	SibSp	Parch	Fare	Embarked	AgeGroup	CabinBool	Title
0	1	0	3	Braund, Mr. Owen Harris	male	22.0	1	0	7.2500	S	Young Adult	0	1
1	2	1	1	Cumings, Mrs. John Bradley (Florence Briggs Th...	female	38.0	1	0	71.2833	C	Senior	1	3
2	3	1	3	Heikkinen, Miss. Laina	female	26.0	0	0	7.9250	S	Adult	0	2
3	4	1	1	Futrelle, Mrs. Jacques Heath (Lily May Peel)	female	35.0	1	0	53.1000	S	Adult	1	3
4	5	0	3	Allen, Mr. William Henry	male	35.0	0	0	8.0500	S	Adult	0	1
5	6	0	3	Moran, Mr. James	male	-0.5	0	0	8.4583	Q	Baby	0	1
6	7	0	1	McCarthy, Mr. Timothy J	male	54.0	0	0	51.8625	S	Senior	1	1
7	8	0	3	Palsson, Master. Gosta Leonard	male	2.0	3	1	21.0750	S	Child	0	4
8	9	1	3	Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)	female	27.0	0	2	11.1333	S	Adult	0	3
9	10	1	2	Nasser, Mrs. Nicholas (Adele Achem)	female	14.0	1	0	30.0708	C	Student	0	3
10	11	1	3	Sandstrom, Miss. Marguerite Rut	female	4.0	1	1	16.7000	S	Child	1	2
11	12	1	1	Bonnell, Miss. Elizabeth	female	58.0	0	0	26.5500	S	Senior	1	2
12	13	0	3	Saundercock, Mr. William Henry	male	20.0	0	0	8.0500	S	Young Adult	0	1
13	14	0	3	Andersson, Mr. Anders Johan	male	39.0	1	5	31.2750	S	Senior	0	1
14	15	0	3	Vestrom, Miss. Hulda Amanda Adolfina	female	14.0	0	0	7.8542	S	Student	0	2
15	16	1	2	Hewlett, Mrs. (Mary D Kingcome)	female	55.0	0	0	16.0000	S	Senior	0	3
16	17	0	3	Rice, Master. Eugene	male	2.0	4	1	29.1250	Q	Child	0	4
17	18	1	2	Williams, Mr. Charles Eugene	male	-0.5	0	0	13.0000	S	Baby	0	1
18	19	0	3	Vander Planke, Mrs. Julius (Emelia Maria Vande...	female	31.0	1	0	18.0000	S	Adult	0	3
19	20	1	3	Masselmani, Mrs. Fatima	female	-0.5	0	0	7.2250	C	Baby	0	3
20	21	0	2	Fynney, Mr. Joseph J	male	35.0	0	0	26.0000	S	Adult	0	1
21	22	1	2	Beesley, Mr. Lawrence	male	34.0	0	0	13.0000	S	Adult	1	1
22	23	1	3	McGowan, Miss. Anna "Annie"	female	15.0	0	0	8.0292	Q	Student	0	2
23	24	1	1	Sloper, Mr. William Thompson	male	28.0	0	0	35.5000	S	Adult	1	1
24	25	0	3	Palsson, Miss. Torborg Danira	female	8.0	3	1	21.0750	S	Teenager	0	2
25	26	1	3	Asplund, Mrs. Carl Oscar (Selma Augusta Emilia...	female	38.0	1	5	31.3875	S	Senior	0	3
26	27	0	3	Emir, Mr. Farred Chehab	male	-0.5	0	0	7.2250	C	Baby	0	1
27	28	0	1	Fortune, Mr. Charles Alexander	male	19.0	3	2	263.0000	S	Young Adult	1	1
28	29	1	3	O'Dwyer, Miss. Ellen "Nellie"	female	-0.5	0	0	7.8792	Q	Baby	0	2
29	30	0	3	Todoroff, Mr. Lalio	male	-0.5	0	0	7.8958	S	Baby	0	1
30	31	0	1	Uruchurtu, Don. Manuel E	male	40.0	0	0	27.7208	C	Senior	0	6
31	32	1	1	Spencer, Mrs. William Augustus (Marie Eugenie)	female	-0.5	1	0	146.5208	C	Baby	1	3
32	33	1	3	Glynn, Miss. Mary Agatha	female	-0.5	0	0	7.7500	Q	Baby	0	2
33	34	0	2	Wheadon, Mr. Edward H	male	66.0	0	0	10.5000	S	Senior	0	1
34	35	0	1	Meyer, Mr. Edgar Joseph	male	28.0	1	0	82.1708	C	Adult	0	1
35	36	0	1	Holverson, Mr. Alexander Oskar	male	42.0	1	0	52.0000	S	Senior	0	1
36	37	1	3	Mamee, Mr. Hanna	male	-0.5	0	0	7.2292	C	Baby	0	1
37	38	0	3	Cann, Mr. Ernest Charles	male	21.0	0	0	8.0500	S	Young Adult	0	1
38	39	0	3	Vander Planke, Miss. Augusta Maria	female	18.0	2	0	18.0000	S	Student	0	2
39	40	1	3	Nicola-Yarred, Miss. Jamila	female	14.0	1	0	11.2417	C	Student	0	2
40	41	0	3	Ahlin, Mrs. Johan (Johanna Persdotter Larsson)	female	40.0	1	0	9.4750	S	Senior	0	3
41	42	0	2	Turpin, Mrs. William John Robert (Dorothy Ann ...	female	27.0	1	0	21.0000	S	Adult	0	3
42	43	0	3	Kraeff, Mr. Theodor	male	-0.5	0	0	7.8958	C	Baby	0	1
43	44	1	2	Laroche, Miss. Simonne Marie Anne Andree	female	3.0	1	2	41.5792	C	Child	0	2
44	45	1	3	Devaney, Miss. Margaret Delia	female	19.0	0	0	7.8792	Q	Young Adult	0	2
45	46	0	3	Rogers, Mr. William John	male	-0.5	0	0	8.0500	S	Baby	0	1
46	47	0	3	Lennon, Mr. Denis	male	-0.5	1	0	15.5000	Q	Baby	0	1
47	48	1	3	O'Driscoll, Miss. Bridget	female	-0.5	0	0	7.7500	Q	Baby	0	2
48	49	0	3	Samaan, Mr. Youssef	male	-0.5	2	0	21.6792	C	Baby	0	1
49	50	0	3	Arnold-Franchi, Mrs. Josef (Josefine Franchi)	female	18.0	1	0	17.8000	S	Student	0	3
# fill missing age with mode age group for each title
mr_age = drt[drt["Title"] == 3]["AgeGroup"].mode() #Adult
miss_age = drt[drt["Title"] == 2]["AgeGroup"].mode() #Student
mrs_age = drt[drt["Title"] == 3]["AgeGroup"].mode() #Adult
master_age = drt[drt["Title"] == 4]["AgeGroup"].mode() #Baby
royal_age = drt[drt["Title"] == 5]["AgeGroup"].mode() #Adult
rare_age = drt[drt["Title"] == 6]["AgeGroup"].mode() #Adult
mr_age = drt[drt["Title"] == 7]["AgeGroup"].mode() #Adult
age_title_mapping = {1: "Young Adult", 2: "Student", 3: "Adult", 4: "Baby", 5: "Adult", 6: "Adult"}
#map each Age value to a numerical value
age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}
drt['AgeGroup'] = drt['AgeGroup'].map(age_mapping)
drt.head(100)
PassengerId	Survived	Pclass	Name	Sex	Age	SibSp	Parch	Fare	Embarked	AgeGroup	CabinBool	Title
0	1	0	3	Braund, Mr. Owen Harris	male	22.0	1	0	7.2500	S	5	0	1
1	2	1	1	Cumings, Mrs. John Bradley (Florence Briggs Th...	female	38.0	1	0	71.2833	C	7	1	3
2	3	1	3	Heikkinen, Miss. Laina	female	26.0	0	0	7.9250	S	6	0	2
3	4	1	1	Futrelle, Mrs. Jacques Heath (Lily May Peel)	female	35.0	1	0	53.1000	S	6	1	3
4	5	0	3	Allen, Mr. William Henry	male	35.0	0	0	8.0500	S	6	0	1
5	6	0	3	Moran, Mr. James	male	-0.5	0	0	8.4583	Q	1	0	1
6	7	0	1	McCarthy, Mr. Timothy J	male	54.0	0	0	51.8625	S	7	1	1
7	8	0	3	Palsson, Master. Gosta Leonard	male	2.0	3	1	21.0750	S	2	0	4
8	9	1	3	Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)	female	27.0	0	2	11.1333	S	6	0	3
9	10	1	2	Nasser, Mrs. Nicholas (Adele Achem)	female	14.0	1	0	30.0708	C	4	0	3
10	11	1	3	Sandstrom, Miss. Marguerite Rut	female	4.0	1	1	16.7000	S	2	1	2
11	12	1	1	Bonnell, Miss. Elizabeth	female	58.0	0	0	26.5500	S	7	1	2
12	13	0	3	Saundercock, Mr. William Henry	male	20.0	0	0	8.0500	S	5	0	1
13	14	0	3	Andersson, Mr. Anders Johan	male	39.0	1	5	31.2750	S	7	0	1
14	15	0	3	Vestrom, Miss. Hulda Amanda Adolfina	female	14.0	0	0	7.8542	S	4	0	2
15	16	1	2	Hewlett, Mrs. (Mary D Kingcome)	female	55.0	0	0	16.0000	S	7	0	3
16	17	0	3	Rice, Master. Eugene	male	2.0	4	1	29.1250	Q	2	0	4
17	18	1	2	Williams, Mr. Charles Eugene	male	-0.5	0	0	13.0000	S	1	0	1
18	19	0	3	Vander Planke, Mrs. Julius (Emelia Maria Vande...	female	31.0	1	0	18.0000	S	6	0	3
19	20	1	3	Masselmani, Mrs. Fatima	female	-0.5	0	0	7.2250	C	1	0	3
20	21	0	2	Fynney, Mr. Joseph J	male	35.0	0	0	26.0000	S	6	0	1
21	22	1	2	Beesley, Mr. Lawrence	male	34.0	0	0	13.0000	S	6	1	1
22	23	1	3	McGowan, Miss. Anna "Annie"	female	15.0	0	0	8.0292	Q	4	0	2
23	24	1	1	Sloper, Mr. William Thompson	male	28.0	0	0	35.5000	S	6	1	1
24	25	0	3	Palsson, Miss. Torborg Danira	female	8.0	3	1	21.0750	S	3	0	2
25	26	1	3	Asplund, Mrs. Carl Oscar (Selma Augusta Emilia...	female	38.0	1	5	31.3875	S	7	0	3
26	27	0	3	Emir, Mr. Farred Chehab	male	-0.5	0	0	7.2250	C	1	0	1
27	28	0	1	Fortune, Mr. Charles Alexander	male	19.0	3	2	263.0000	S	5	1	1
28	29	1	3	O'Dwyer, Miss. Ellen "Nellie"	female	-0.5	0	0	7.8792	Q	1	0	2
29	30	0	3	Todoroff, Mr. Lalio	male	-0.5	0	0	7.8958	S	1	0	1
...	...	...	...	...	...	...	...	...	...	...	...	...	...
70	71	0	2	Jenkin, Mr. Stephen Curnow	male	32.0	0	0	10.5000	S	6	0	1
71	72	0	3	Goodwin, Miss. Lillian Amy	female	16.0	5	2	46.9000	S	4	0	2
72	73	0	2	Hood, Mr. Ambrose Jr	male	21.0	0	0	73.5000	S	5	0	1
73	74	0	3	Chronopoulos, Mr. Apostolos	male	26.0	1	0	14.4542	C	6	0	1
74	75	1	3	Bing, Mr. Lee	male	32.0	0	0	56.4958	S	6	0	1
75	76	0	3	Moen, Mr. Sigurd Hansen	male	25.0	0	0	7.6500	S	6	1	1
76	77	0	3	Staneff, Mr. Ivan	male	-0.5	0	0	7.8958	S	1	0	1
77	78	0	3	Moutal, Mr. Rahamin Haim	male	-0.5	0	0	8.0500	S	1	0	1
78	79	1	2	Caldwell, Master. Alden Gates	male	1.0	0	2	29.0000	S	2	0	4
79	80	1	3	Dowdell, Miss. Elizabeth	female	30.0	0	0	12.4750	S	6	0	2
80	81	0	3	Waelens, Mr. Achille	male	22.0	0	0	9.0000	S	5	0	1
81	82	1	3	Sheerlinck, Mr. Jan Baptist	male	29.0	0	0	9.5000	S	6	0	1
82	83	1	3	McDermott, Miss. Brigdet Delia	female	-0.5	0	0	7.7875	Q	1	0	2
83	84	0	1	Carrau, Mr. Francisco M	male	28.0	0	0	47.1000	S	6	0	1
84	85	1	2	Ilett, Miss. Bertha	female	17.0	0	0	10.5000	S	4	0	2
85	86	1	3	Backstrom, Mrs. Karl Alfred (Maria Mathilda Gu...	female	33.0	3	0	15.8500	S	6	0	3
86	87	0	3	Ford, Mr. William Neal	male	16.0	1	3	34.3750	S	4	0	1
87	88	0	3	Slocovski, Mr. Selman Francis	male	-0.5	0	0	8.0500	S	1	0	1
88	89	1	1	Fortune, Miss. Mabel Helen	female	23.0	3	2	263.0000	S	5	1	2
89	90	0	3	Celotti, Mr. Francesco	male	24.0	0	0	8.0500	S	5	0	1
90	91	0	3	Christmann, Mr. Emil	male	29.0	0	0	8.0500	S	6	0	1
91	92	0	3	Andreasson, Mr. Paul Edvin	male	20.0	0	0	7.8542	S	5	0	1
92	93	0	1	Chaffee, Mr. Herbert Fuller	male	46.0	1	0	61.1750	S	7	1	1
93	94	0	3	Dean, Mr. Bertram Frank	male	26.0	1	2	20.5750	S	6	0	1
94	95	0	3	Coxon, Mr. Daniel	male	59.0	0	0	7.2500	S	7	0	1
95	96	0	3	Shorney, Mr. Charles Joseph	male	-0.5	0	0	8.0500	S	1	0	1
96	97	0	1	Goldschmidt, Mr. George B	male	71.0	0	0	34.6542	C	7	1	1
97	98	1	1	Greenfield, Mr. William Bertram	male	23.0	0	1	63.3583	C	5	1	1
98	99	1	2	Doling, Mrs. John T (Ada Julia Bone)	female	34.0	0	1	23.0000	S	6	0	3
99	100	0	2	Kantor, Mr. Sinai	male	34.0	1	0	26.0000	S	6	0	1
100 rows × 13 columns

#drop the name feature since it contains no more useful information.
drt = drt.drop(['Name'], axis = 1)
#map each Sex value to a numerical value
sex_mapping = {"male": 0, "female": 1}
drt['Sex'] = drt['Sex'].map(sex_mapping)
drt.head()
PassengerId	Survived	Pclass	Sex	Age	SibSp	Parch	Fare	Embarked	AgeGroup	CabinBool	Title
0	1	0	3	0	22.0	1	0	7.2500	S	5	0	1
1	2	1	1	1	38.0	1	0	71.2833	C	7	1	3
2	3	1	3	1	26.0	0	0	7.9250	S	6	0	2
3	4	1	1	1	35.0	1	0	53.1000	S	6	1	3
4	5	0	3	0	35.0	0	0	8.0500	S	6	0	1
#map each Embarked values to a numerical value
embarked_mapping = {"S": 1, "C": 2, "Q": 3}
drt['Embarked'] = drt['Embarked'].map(embarked_mapping)

drt.head()
PassengerId	Survived	Pclass	Sex	Age	SibSp	Parch	Fare	Embarked	AgeGroup	CabinBool	Title
0	1	0	3	0	22.0	1	0	7.2500	1	5	0	1
1	2	1	1	1	38.0	1	0	71.2833	2	7	1	3
2	3	1	3	1	26.0	0	0	7.9250	1	6	0	2
3	4	1	1	1	35.0	1	0	53.1000	1	6	1	3
4	5	0	3	0	35.0	0	0	8.0500	1	6	0	1
#map Fare values to numerical values
drt['FareClass'] = pd.qcut(drt['Fare'], 4, labels = [1, 2, 3, 4])
drt.head()
PassengerId	Survived	Pclass	Sex	Age	SibSp	Parch	Fare	Embarked	AgeGroup	CabinBool	Title	FareClass
0	1	0	3	0	22.0	1	0	7.2500	1	5	0	1	1
1	2	1	1	1	38.0	1	0	71.2833	2	7	1	3	4
2	3	1	3	1	26.0	0	0	7.9250	1	6	0	2	2
3	4	1	1	1	35.0	1	0	53.1000	1	6	1	3	4
4	5	0	3	0	35.0	0	0	8.0500	1	6	0	1	2
#drop Fare values
drt = drt.drop(['Fare'], axis = 1)

drt.head()
PassengerId	Survived	Pclass	Sex	Age	SibSp	Parch	Embarked	AgeGroup	CabinBool	Title	FareClass
0	1	0	3	0	22.0	1	0	1	5	0	1	1
1	2	1	1	1	38.0	1	0	2	7	1	3	4
2	3	1	3	1	26.0	0	0	1	6	0	2	2
3	4	1	1	1	35.0	1	0	1	6	1	3	4
4	5	0	3	0	35.0	0	0	1	6	0	1	2
#Drop age
drt = drt.drop(['Age'], axis = 1)

drt.head()
PassengerId	Survived	Pclass	Sex	SibSp	Parch	Embarked	AgeGroup	CabinBool	Title	FareClass
0	1	0	3	0	1	0	1	5	0	1	1
1	2	1	1	1	1	0	2	7	1	3	4
2	3	1	3	1	0	0	1	6	0	2	2
3	4	1	1	1	1	0	1	6	1	3	4
4	5	0	3	0	0	0	1	6	0	1	2
drt.isnull().sum().sort_index()/len(drt)
AgeGroup       0.0
CabinBool      0.0
Embarked       0.0
FareClass      0.0
Parch          0.0
PassengerId    0.0
Pclass         0.0
Sex            0.0
SibSp          0.0
Survived       0.0
Title          0.0
dtype: float64
drt.head(100)
PassengerId	Survived	Pclass	Sex	SibSp	Parch	Embarked	AgeGroup	CabinBool	Title	FareClass
0	1	0	3	0	1	0	1	5	0	1	1
1	2	1	1	1	1	0	2	7	1	3	4
2	3	1	3	1	0	0	1	6	0	2	2
3	4	1	1	1	1	0	1	6	1	3	4
4	5	0	3	0	0	0	1	6	0	1	2
5	6	0	3	0	0	0	3	1	0	1	2
6	7	0	1	0	0	0	1	7	1	1	4
7	8	0	3	0	3	1	1	2	0	4	3
8	9	1	3	1	0	2	1	6	0	3	2
9	10	1	2	1	1	0	2	4	0	3	3
10	11	1	3	1	1	1	1	2	1	2	3
11	12	1	1	1	0	0	1	7	1	2	3
12	13	0	3	0	0	0	1	5	0	1	2
13	14	0	3	0	1	5	1	7	0	1	4
14	15	0	3	1	0	0	1	4	0	2	1
15	16	1	2	1	0	0	1	7	0	3	3
16	17	0	3	0	4	1	3	2	0	4	3
17	18	1	2	0	0	0	1	1	0	1	2
18	19	0	3	1	1	0	1	6	0	3	3
19	20	1	3	1	0	0	2	1	0	3	1
20	21	0	2	0	0	0	1	6	0	1	3
21	22	1	2	0	0	0	1	6	1	1	2
22	23	1	3	1	0	0	3	4	0	2	2
23	24	1	1	0	0	0	1	6	1	1	4
24	25	0	3	1	3	1	1	3	0	2	3
25	26	1	3	1	1	5	1	7	0	3	4
26	27	0	3	0	0	0	2	1	0	1	1
27	28	0	1	0	3	2	1	5	1	1	4
28	29	1	3	1	0	0	3	1	0	2	1
29	30	0	3	0	0	0	1	1	0	1	1
...	...	...	...	...	...	...	...	...	...	...	...
70	71	0	2	0	0	0	1	6	0	1	2
71	72	0	3	1	5	2	1	4	0	2	4
72	73	0	2	0	0	0	1	5	0	1	4
73	74	0	3	0	1	0	2	6	0	1	2
74	75	1	3	0	0	0	1	6	0	1	4
75	76	0	3	0	0	0	1	6	1	1	1
76	77	0	3	0	0	0	1	1	0	1	1
77	78	0	3	0	0	0	1	1	0	1	2
78	79	1	2	0	0	2	1	2	0	4	3
79	80	1	3	1	0	0	1	6	0	2	2
80	81	0	3	0	0	0	1	5	0	1	2
81	82	1	3	0	0	0	1	6	0	1	2
82	83	1	3	1	0	0	3	1	0	2	1
83	84	0	1	0	0	0	1	6	0	1	4
84	85	1	2	1	0	0	1	4	0	2	2
85	86	1	3	1	3	0	1	6	0	3	3
86	87	0	3	0	1	3	1	4	0	1	4
87	88	0	3	0	0	0	1	1	0	1	2
88	89	1	1	1	3	2	1	5	1	2	4
89	90	0	3	0	0	0	1	5	0	1	2
90	91	0	3	0	0	0	1	6	0	1	2
91	92	0	3	0	0	0	1	5	0	1	1
92	93	0	1	0	1	0	1	7	1	1	4
93	94	0	3	0	1	2	1	6	0	1	3
94	95	0	3	0	0	0	1	7	0	1	1
95	96	0	3	0	0	0	1	1	0	1	2
96	97	0	1	0	0	0	2	7	1	1	4
97	98	1	1	0	0	1	2	5	1	1	4
98	99	1	2	1	0	1	1	6	0	3	3
99	100	0	2	0	1	0	1	6	0	1	3
100 rows × 11 columns

Splitting drt on 80% for training and 20% for testing Testing Different Models
I will be testing the data using Gaussian Naive Bayes, Logistic Regression, Support Vector Machines, Perceptron,
Decision Tree Classifier, Random Forest Classifier, k-Nearest Neighbors (KNN), Stochastic Gradient Descent,
Gradient Boosting Classifier
from sklearn.model_selection import train_test_split

predictors = drt.drop(['Survived', 'PassengerId'], axis=1)
target = drt["Survived"]
x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.2, random_state = 0)
# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gaussian)
79.33
# Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_logreg)
78.77
# Support Vector Machines
from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_val)
acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_svc)
82.12
# Linear SVC
from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_val)
acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_linear_svc)
77.09
# Perceptron
from sklearn.linear_model import Perceptron

perceptron = Perceptron()
perceptron.fit(x_train, y_train)
y_pred = perceptron.predict(x_val)
acc_perceptron = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_perceptron)
77.65
#Decision Tree
from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_decisiontree)
81.01
# Random Forest
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_randomforest)
84.92
# KNN or k-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_knn)
81.01
# Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_val)
acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_sgd)
76.54
# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_val)
acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gbk)
82.68
Now comparing the accuracy of each model
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 'Linear SVC', 
              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier'],
    'Score': [acc_svc, acc_knn, acc_logreg, 
              acc_randomforest, acc_gaussian, acc_perceptron,acc_linear_svc, acc_decisiontree,
              acc_sgd, acc_gbk]})
models.sort_values(by='Score', ascending=False)
Model	Score
3	Random Forest	84.92
9	Gradient Boosting Classifier	82.68
0	Support Vector Machines	82.12
1	KNN	81.01
7	Decision Tree	81.01
4	Naive Bayes	79.33
2	Logistic Regression	78.77
5	Perceptron	77.65
6	Linear SVC	77.09
8	Stochastic Gradient Descent	76.54
