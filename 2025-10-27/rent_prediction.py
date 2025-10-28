import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# ダミーデータ作成
data = pd.DataFrame({
    'area': [20, 30, 35, 40, 45, 50, 55, 60],
    'distance': [15, 13, 10, 9, 8, 6, 4, 2],
    'age': [20, 15, 10, 7, 5, 4, 3, 2],
    'rent': [50000, 70000, 90000, 100000, 110000, 113000, 121000, 130000]
})


X = data[['area', 'distance', 'age']]
y = data['rent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train:\n", X_train)
print("y_train:\n", y_train)
print("X_test:\n", X_test)
print("y_test:\n", y_test)


model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("予測:", predictions)
print("MSE:", mean_squared_error(y_test, predictions))
print("R²:", r2_score(y_test, predictions))
print("相関関数:", data.corr())




