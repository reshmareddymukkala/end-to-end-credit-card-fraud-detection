import joblib
import pandas as pd
import numpy as np

model = joblib.load("models/lightgbm.pkl")
df = pd.read_csv("data/features.csv")
X = df.drop(columns=["Class"])
y = df["Class"]

# Take a small sample with known fraud and known legit
fraud_rows = X[y == 1].head(5)
legit_rows = X[y == 0].head(5)

print("Predictions on 5 fraud transactions:")
print(model.predict_proba(fraud_rows))
print("\nPredictions on 5 legit transactions:")
print(model.predict_proba(legit_rows))