import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# ダミーデータ作成
data = pd.DataFrame({
    'area': [20, 30, 40, 50, 60],
    'distance': [10, 8, 6, 4, 2],
    'age': [20, 15, 10, 5, 2],
    'rent': [50000, 70000, 90000, 110000, 130000]
})


X = data[['area', 'distance', 'age']]
y = data['rent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("予測:", predictions)
print("MSE:", mean_squared_error(y_test, predictions))




