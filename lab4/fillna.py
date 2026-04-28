import pandas as pd

df = pd.read_csv("data/titanic.csv")
df["Age"] = df["Age"].fillna(df["Age"].mean())

df.to_csv("data/titanic_modified.csv", index=False)
