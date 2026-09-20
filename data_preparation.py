
import pandas as pd

print("Data Preparation Started")

df = pd.read_csv("SuperKart.csv")

df.drop(columns=["Product_Id", "Store_Id"], inplace=True)

print("Data Preparation Completed")
print(df.head())
