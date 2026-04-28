import pandas as pd

df = pd.read_csv("data/titanic_modified.csv")
df = pd.get_dummies(df, columns=["Sex"], prefix="Sex", drop_first=False)

df["Sex_male"] = df["Sex_male"].astype(int)
df["Sex_female"] = df["Sex_female"].astype(int)

df.to_csv("data/titanic_modified.csv", index=False)