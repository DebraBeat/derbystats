import pandas as pd
import datetime

df = pd.read_csv('GameData.csv')
print(df['Home Team'].head)
print(df['Away Team'].head)

# While we've converted our data from a number of excel sheets to a dataFrame, it still needs to be cleaned.
# First we drop this unnecessary column.
df = df.drop(columns='Unnamed: 0')
# We want to make our team name columns string, our date column datetime, and the rest integers
# First let's start with the team names and date columns
df['Home Team'] = df['Home Team'].astype(str)
df['Away Team'] = df['Away Team'].astype(str)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

numeric_cols = df.columns.drop(['Home Team', 'Away Team', 'Date'])
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

print(df['Home Team Score'].head)
print(df['Away Team Score'].head)