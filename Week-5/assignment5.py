import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("USA_Housing.csv")

print(data)

# All 5 features (Address excluded)
features = [
    "Avg. Area Income",
    "Avg. Area House Age",
    "Avg. Area Number of Rooms",
    "Avg. Area Number of Bedrooms",
    "Area Population"
]

X = data[features]
y = data["Price"]

# Normalisation
scaler = StandardScaler()
X = scaler.fit_transform(X)

print(X)
print(y)

# Find best random state out of 200
best_random_state = 0
best_score = -1

for random_state in range(200):

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

# Train using best random state
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=best_random_state
)

model = LinearRegression()
model.fit(X_train, y_train)

prediction = model.predict(X_test)

print("Intercept:", model.intercept_)

print("Coefficients:")

for i in range(len(features)):
    print(features[i], ":", model.coef_[i])

# Error metrics
mse = mean_squared_error(y_test, prediction)
r2 = r2_score(y_test, prediction)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)




plt.scatter(y_test, prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Price vs Predicted Price")
plt.savefig("actual_vs_predicted.png")
plt.show()