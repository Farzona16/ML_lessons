import pandas as pd
import seaborn as sn
df=sn.load_dataset("titanic")
print(df.shape)
print(df.columns)
print(df.first)
print(df.columns.dtype)
df.info()
df.dtypes

nums=df.select_dtypes(include=['int64','float64'])
mean_df=nums.mean()
print(mean_df)
med_df=nums.median()
print(med_df)
mode_df=nums.mode().iloc[0]
print(mode_df)
print(df['sex'].nunique())
print(df['embarked'].nunique())
print(df.groupby("sex")["survived"].mean())
print(df.groupby("pclass")["fare"].mean())
pd.pivot_table(df,values='survived',index='sex',columns='pclass', aggfunc='mean')

print(df.isnull().sum().sort_values(ascending=False))

df['age'].fillna({'age':df['age'].mean()},inplace=True)
df['embarked'].fillna('S',inplace=True)
df['embark_town'].fillna('Southampton',inplace=True)
df['deck'].fillna(df['deck'].mode().iloc[0],inplace=True)
print(df.isnull().sum().sort_values(ascending=False))

import matplotlib.pyplot as plt
sn.histplot(df['age'], kde=True)
plt.show()
sn.countplot(x='sex',data=df)
plt.show()

sn.barplot(x='survived',y='sex',data=df)
plt.show()

sn.barplot(x='sex',y='survived',data=df)
plt.ylabel("Survival rate")
plt.show()

sn.barplot(x="pclass",y='survived',hue='sex',data=df)
plt.ylabel("Survival rate")
plt.show()

pivot_ps=pd.pivot_table(df,values='survived',index='pclass',columns='sex',aggfunc='mean')

sn.heatmap(pivot_ps,annot=True,cmap='coolwarm')
plt.title("Survival rate v=by pclass and sex")
plt.show()

df.duplicated().sum()

df.drop_duplicates(inplace=True)

pd.get_dummies(df,columns=['embarked'],drop_first=True)

df.isnull().sum()
df['sex']=df['sex'].map({'male':0,'female':1})
df['family_size']=df['sibsp']+df['parch']+1
df['isalone']=(df['family_size']==1).astype(int)
print(df['family_size'],df['isalone'])

sn.pairplot(
    df[['age','fare','family_size','survived']],
    hue='survived'
)
plt.show()

corr=df[['age','fare','family_size','survived']].corr()
sn.heatmap(corr,annot=True,cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()