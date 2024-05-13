import pandas as pd
import seaborn as sns
import matplotlib
from CleanDataframe import clean

df = pd.read_csv('GameData.csv')
df = clean(df)

sns.set_theme()
