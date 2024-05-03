import pandas as pd

df = pd.read_csv('GameData.csv')
df = df.drop(columns='Unnamed: 0')

