Python 3.12.7 (tags/v3.12.7:0b05ead, Oct  1 2024, 03:06:41) [MSC v.1941 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('C:/Users/jenny/Downloads/titanic/train.csv') #데이터 불러오기
df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']] #열 선택
     Survived  Pclass     Sex   Age     Fare
0           0       3    male  22.0   7.2500
1           1       1  female  38.0  71.2833
2           1       3  female  26.0   7.9250
3           1       1  female  35.0  53.1000
4           0       3    male  35.0   8.0500
..        ...     ...     ...   ...      ...
886         0       2    male  27.0  13.0000
887         1       1  female  19.0  30.0000
888         0       3  female   NaN  23.4500
889         1       1    male  26.0  30.0000
890         0       3    male  32.0   7.7500

[891 rows x 5 columns]
df[df['Age'] >= 20] #나이 20세 이상인 사람만 선택
     PassengerId  Survived  Pclass  ...     Fare Cabin  Embarked
0              1         0       3  ...   7.2500   NaN         S
1              2         1       1  ...  71.2833   C85         C
2              3         1       3  ...   7.9250   NaN         S
3              4         1       1  ...  53.1000  C123         S
4              5         0       3  ...   8.0500   NaN         S
..           ...       ...     ...  ...      ...   ...       ...
884          885         0       3  ...   7.0500   NaN         S
885          886         0       3  ...  29.1250   NaN         Q
886          887         0       2  ...  13.0000   NaN         S
889          890         1       1  ...  30.0000  C148         C
890          891         0       3  ...   7.7500   NaN         Q

[550 rows x 12 columns]
train_data = pd.read_csv('C:/Users/jenny/Downloads/titanic/train.csv')
import math
def age_categorize(age): #나이 범주형으로 변환
    if math.isnan(age):
        return -1
    return math.floor(age / 10) * 10

train_data['Age'].apply(age_categorize)
0      20
1      30
2      20
3      30
4      30
       ..
886    20
887    10
888    -1
889    20
890    30
Name: Age, Length: 891, dtype: int64
df['age_categorize2'] = df['Age'].apply(lambda x: 'Child' if x < 20 else 'Adult') #아이와 성인 구분 및 새로운 열 추가
print(df)
     PassengerId  Survived  Pclass  ... Cabin Embarked  age_categorize2
0              1         0       3  ...   NaN        S            Adult
1              2         1       1  ...   C85        C            Adult
2              3         1       3  ...   NaN        S            Adult
3              4         1       1  ...  C123        S            Adult
4              5         0       3  ...   NaN        S            Adult
..           ...       ...     ...  ...   ...      ...              ...
886          887         0       2  ...   NaN        S            Adult
887          888         1       1  ...   B42        S            Child
888          889         0       3  ...   NaN        S            Adult
889          890         1       1  ...  C148        C            Adult
890          891         0       3  ...   NaN        Q            Adult

[891 rows x 13 columns]
import numpy as np
df['age_categorize3'] = df['Age'].apply(lambda x: 'Child' if x < 20 and x != -1 else ('Adult' if x >= 20 else np.nan)) #범주형 변환 과정에서 -1 값인 데이터 처리
print(df)
     PassengerId  Survived  Pclass  ... Embarked age_categorize2  age_categorize3
0              1         0       3  ...        S           Adult            Adult
1              2         1       1  ...        C           Adult            Adult
2              3         1       3  ...        S           Adult            Adult
3              4         1       1  ...        S           Adult            Adult
4              5         0       3  ...        S           Adult            Adult
..           ...       ...     ...  ...      ...             ...              ...
886          887         0       2  ...        S           Adult            Adult
887          888         1       1  ...        S           Child            Child
888          889         0       3  ...        S           Adult              NaN
889          890         1       1  ...        C           Adult            Adult
890          891         0       3  ...        Q           Adult            Adult

[891 rows x 14 columns]
>>> df.isnull().sum() #결측치 확인
PassengerId          0
Survived             0
Pclass               0
Name                 0
Sex                  0
Age                177
SibSp                0
Parch                0
Ticket               0
Fare                 0
Cabin              687
Embarked             2
age_categorize2      0
age_categorize3    177
dtype: int64
>>> df['Age'].fillna(df['Age'].mean()) #결측치 평균값으로 채워넣기
0      22.000000
1      38.000000
2      26.000000
3      35.000000
4      35.000000
         ...    
886    27.000000
887    19.000000
888    29.699118
889    26.000000
890    32.000000
Name: Age, Length: 891, dtype: float64
>>> sns.barplot(x='Age', y='Fare', data=df) #나이와 요금의 분포 히스토그램 시각화
<Axes: xlabel='Age', ylabel='Fare'>
>>> plt.show()
>>> sns.stripplot(data=df, x="Sex", y="Survived") #성별에 따른 생존율 산점도 시각화
<Axes: xlabel='Sex', ylabel='Survived'>
>>> plt.show()
