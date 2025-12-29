import os
import tarfile
import urllib.request
import pandas as pd

DOWNLOAD_ROOT = "https://github.com/ageron/data/raw/main/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    with tarfile.open(tgz_path) as housing_tgz:
        housing_tgz.extractall(path=housing_path)

fetch_housing_data()
df = pd.read_csv("datasets/housing/housing/housing.csv")

df.info()
df.first
df.describe()
df['ocean_proximity'].value_counts()
df.isnull().sum()
df.select_dtypes(include=['int64','float64']).columns
df.select_dtypes(include=['object']).columns

import matplotlib.pyplot as plt
df.boxplot(column='population')
plt.show()

df['total_rooms'].hist(bins=50)
plt.show()

Q1=df['population'].quantile(0.25)
Q3=df['population'].quantile(0.75)
IQR=Q3-Q1
outliers=df[(df['population']<Q1-1.5*IQR)|(df['population']>Q3+1.5*IQR)]

print(outliers)
df.isna().sum()
median=df['total_bedrooms'].median()
df['total_bedrooms'].fillna(median, inplace=True)

def missing_report(df):
    missing_count=df.isna().sum()
    missing_percent=(missing_count/len(df))*100
    report=(
        pd.DataFrame({
            "column":missing_count.index,
            "missing_count": missing_count.values,
            "missing_percent":missing_percent.values
        })
        .query('missing_count>0')
        .reset_index(drop=True)
    )
    return report
missing_report(df)

pd.get_dummies(df['ocean_proximity']).astype(int)

ocean_encoded=pd.get_dummies(df['ocean_proximity'], dtype=int)

df_encoded=pd.concat(
    [df.drop('ocean_proximity', axis=1), ocean_encoded],
    axis=1

)
print(df_encoded)
df.describe()

import numpy as np
from sklearn.preprocessing import StandardScaler
df["total_rooms_log"]=np.log1p(df['total_rooms'])
scaler=StandardScaler()
df['total_rooms_scaled']=scaler.fit_transform(
    df[['total_rooms_log']]
)

df["population_log"]=np.log1p(df['population'])
scaler=StandardScaler()
df['population_scaled']=scaler.fit_transform(
    df[['population_log']]
)

df["households_log"]=np.log1p(df['households'])
scaler=StandardScaler()
df['households_scaled']=scaler.fit_transform(
    df[['households_log']]
)

df["total_bedrooms_log"]=np.log1p(df['total_bedrooms'])
scaler=StandardScaler()
df['totat_bedrooms_scaled']=scaler.fit_transform(
    df[['total_bedrooms_log']]
)

df['total_rooms'].hist(bins=50)
plt.title("Original total_rooms")
plt.show()

df['total_rooms_log'].hist(bins=50)
plt.title("Original total_rooms")
plt.show()

df['total_rooms_scaled'].hist(bins=50)
plt.title("Original total_rooms")
plt.show()

df['rooms_per_household']=df['total_rooms']/df['households']
df['bedrooms_per_room']=df['total_bedrooms']/df['total_rooms']
df['population_per_household']=df['population']/df['households']