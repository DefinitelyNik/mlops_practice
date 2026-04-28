import pandas as pd

df = pd.read_csv("data/titanic.csv")
df = df[["Pclass", "Sex", "Age"]]

df.to_csv("data/titanic.csv", index=False)