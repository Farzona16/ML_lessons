import numpy as np
import pandas as pd

df=pd.read_csv('data/housing.csv')
df.info()
df.shape
df.first

import matplotlib.pyplot as plt
plt.scatter(df['area'],df["price"])
plt.xlabel('area (kv.m)')
plt.ylabel('price')
plt.title('area vs price')
plt.show()

df.isnull().sum()

plt.boxplot(df['price'])
plt.title('price outliers')
plt.show()

plt.boxplot(df["area"])
plt.title('area outliers')
plt.show()

df.describe()

df['log_price']=np.log(df['price'])
df['log_area']=np.log(df['area'])

plt.scatter(df['log_area'],df['log_price'])
plt.xlabel('log(area)')
plt.ylabel('log(price)')
plt.title('log(area) vs log(price)')
plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X=df[['log_area']]
y=df['log_price']
X_train, X_test, y_train, y_test=train_test_split(
    X,y, test_size=0.2, random_state=42
)
model=LinearRegression()
model.fit(X,y) #type: ignore
beta_0=model.intercept_
beta_1=model.coef_[0]

print(beta_0, beta_1)

x=X_train['log_area'].values
y=y_train.values
x_mean=np.mean(x)
y_mean=np.mean(y)
m=np.sum((x-x_mean)*(y-y_mean))/np.sum((x-x_mean)**2)
b=y_mean-m*x_mean
print(m,b)

plt.scatter(df['log_area'], df['log_price'], alpha=0.4)
plt.plot(df['log_area'], model.predict(df[['log_area']]), color='red')
plt.xlabel('log(area)')
plt.ylabel('log(price)')
plt.title('log-log linear regression')
plt.show()

y_test_pred=model.predict(X_test)

from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
mae=mean_absolute_error(y_test,y_test_pred)
mse=mean_squared_error(y_test,y_test_pred)
r2=r2_score(y_test, y_test_pred)
print(mae,mse,r2)

residuals=y_test-y_test_pred
plt.scatter(y_test_pred,residuals)
plt.axhline(0,color='red')
plt.xlabel('predicted log(price)')
plt.ylabel('residuals')
plt.title('residuals vs predicted')
plt.show()

area_1000=np.log(1000)
log_price_pred=model.predict([[area_1000]]) #type:ignore
price_pred=np.exp(log_price_pred)
print(price_pred)