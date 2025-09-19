import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
df=sn.load_dataset("titanic")
# print(df.columns)
# print(df.head(10))
# print(df.shape)
# print(df.dtypes)
# print(df.count())

# nums=df.select_dtypes(include=['int64','float64'])
# means_df=nums.mean()
# med_df=nums.median()
# mode_df=nums.mode().iloc[0]
# print(means_df)
# print(med_df)
# print(mode_df)
# print(df['sex'].value_counts())
# print(df['embarked'].value_counts())
# print(df.groupby("sex")['survived'].mean())
# print(df.groupby('pclass')['fare'].mean())
# print(pd.pivot_table(df,values='survived',index='sex', columns='pclass', aggfunc='mean'))
# print(df.isnull().sum())
# print(df['sex'].nunique())
# print(df['embarked'].nunique())

# missing_vals=df.isnull().sum()
# most_missing=missing_vals.sort_values(ascending=False)
# print(most_missing)

# df_cleaned=df.dropna()
# # df['age'].fillna(df['age'].mean(),inplace=True)
# df.fillna({'age':df['age'].mean()},inplace=True)

# plt.figure(figsize=(12,5))
# plt.subplot(1,2,1)
# sn.histplot(df["age"].dropna(),bins=30, kde=True, color="skyblue")
# plt.title("Age distribution (hist)")

# plt.subplot(1,2,2)
# sn.boxplot(x=df["age"],color="lightgreen")
# plt.title("age distribution (boxplot)")
# plt.show()

# sn.countplot(x="sex", data=df,palette="Set2")
# plt.title("sex distribution")
# plt.show()

# plt.figure(figsize=(12,5))
# plt.subplot(1,2,1)
# df['survived'].value_counts().plot.pie(autopct="%1.1f%%", colors=['lightcoral','lightblue'], labels=['Died','Survived'])
# plt.title("survival dist pie chart")

# plt.subplot(1,2,2)
# sn.countplot(x="survived", data=df, palette="coolwarm")
# plt.title("survival dist bar plot ")
# plt.show()

# sn.barplot(x='pclass',y='survived',hue='sex', data=df, palette='muted')

# sn.barplot(x='embarked', y='survived', hue='sex', data=df, palette='Set1')

# pivot=df.pivot_table(index='sex', columns='pclass', values="survived", aggfunc='mean')
# sn.heatmap(pivot, annot=True, cmap='YlGnBu', fmt='.02f')

# plt.show()


print(df.shape)

df=df.drop_duplicates()
print(df.shape)
df['age']=df['age'].fillna(df['age'].median())
df['embarked']=df['embarked'].fillna(df['embarked'].mode()[0])

df['sex']=df['sex'].map({'male':0, 'female':1})
df=pd.get_dummies(df,columns=['embarked'], drop_first=True)
print(df.isnull().sum())
print(df.head())