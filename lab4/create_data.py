from catboost.datasets import titanic

train_df, _ = titanic()
train_df.to_csv("data/titanic.csv", index=False)