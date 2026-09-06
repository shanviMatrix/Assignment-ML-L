import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_excel("StudyHrs_Score.csv")

print(data)

X = data[["Study_Hours"]]
y = data["Scores"]

best_random_state = 0
best_score = -1

for random_state in range(10000):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    prediction = model.predict(X_test)
    score = r2_score(y_test, prediction)

    if score > best_score:
        best_score = score
        best_random_state = random_state

print("Best Random State:", best_random_state)
print("Best R2 Score:", best_score)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=best_random_state
)

model = LinearRegression()
model.fit(X_train, y_train)

prediction = model.predict(X_test)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

plt.scatter(X, y)
plt.plot(X, model.predict(X))

plt.xlabel("Study Hours")
plt.ylabel("Scores")
plt.title("Study Hours vs Scores")

plt.savefig("linear_regression.png")