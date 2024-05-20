import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from CleanDataframe import clean

df = pd.read_csv('GameData.csv')
df = clean(df)

jam_cols = []

for s in list(df.columns):
    if 'Jam' in s:
        jam_cols.append(s)

jam_df = df[jam_cols]
home_jam_df = jam_df.iloc[:, 0:78]

sns.set_theme()
tips = sns.load_dataset("tips")

# sns.relplot(
#     # data=home_jam_df,
#     x=pd.Series(home_jam_df.columns),
#     y=home_jam_df.iloc[:, :].mean()
# )
plt.show()
