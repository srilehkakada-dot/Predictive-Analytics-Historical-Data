import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("historical_sales.csv")

# Convert months to numbers
df["Month_Num"] = range(1, len(df)+1)

# Features and target
X = df[["Month_Num"]]
y = df["Sales"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Predictions
predictions = model.predict(X)

# Accuracy
score = r2_score(y, predictions)
print("Model Accuracy (R² Score):", round(score, 2))

# Forecast next 3 months
future_months = pd.DataFrame({"Month_Num":[13,14,15]})
future_predictions = model.predict(future_months)

print("\nFuture Forecast:")
for i, pred in enumerate(future_predictions, start=13):
    print(f"Month {i}: {round(pred,2)}")

# Visualization
plt.figure(figsize=(8,5))
plt.plot(df["Month_Num"], y, marker='o', label="Actual Sales")
plt.plot(df["Month_Num"], predictions, label="Predicted Sales")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Sales Forecasting using Linear Regression")
plt.legend()

plt.savefig("sales_forecast.png")
plt.show()