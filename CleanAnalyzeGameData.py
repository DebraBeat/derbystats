import pandas as pd
import datetime
import numpy as np

df = pd.read_csv('GameData.csv')

# While we've converted our data from a number of excel sheets to a dataFrame, it still needs to be cleaned.
# First we drop this unnecessary column.
df = df.drop(columns='Unnamed: 0')
# We want to make our team name columns string, our date column datetime, and the rest integers
# First let's start with the team names and date columns
df['Home Team'] = df['Home Team'].astype(str)
df['Away Team'] = df['Away Team'].astype(str)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Convert values in numeric cols to integers
numeric_cols = df.columns.drop(['Home Team', 'Away Team', 'Date'])
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Create a primary scores DataFrame and drop indices with missing values.
primary_scores_df = df[['Home Team', 'Away Team', 'Date', 'Home Team Score', 'Away Team Score']].dropna()
# Sort by date
primary_scores_df = primary_scores_df.sort_values(by=['Date'])
# ELO Ranking System
# First we create a DataFrame for each team and their ELO Score
elo_df1 = pd.DataFrame(primary_scores_df['Home Team'])
elo_df2 = pd.DataFrame(primary_scores_df['Away Team'])

elo_df1.rename({'Home Team': 'Team'})
elo_df2.rename({'Away Team': 'Team'})
# Save the primary scores df for later use
primary_scores_df.to_csv('PrimaryScoresData')