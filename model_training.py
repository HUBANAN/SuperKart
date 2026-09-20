
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

print("Model Training Started")

df = pd.read_csv("SuperKart.csv")

X = df.drop(columns=["Product_Store_Sales_Total"])
y = df["Product_Store_Sales_Total"]

Xtrain, Xtest, ytrain, ytest = train_test_split(
X,
y,
test_size=0.2,
random_state=42
)

model = RandomForestRegressor(
n_estimators=100,
random_state=42
)

model.fit(Xtrain, ytrain)

joblib.dump(
model,
"best_superkart_model.pkl"
)

print("Model Training Completed")
