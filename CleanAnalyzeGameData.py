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


# Now we need two methods: one for expected score, and the other for updating ranking
# Expected score method
class Elo:
    def __init__(self):
        # Number comes from wikipedia, could be changed later
        k = 32
        # Create a DataFrame for each team and their ELO Score
        self.elo_df = pd.DataFrame()

    def create_elo_df(self, df: pd.DataFrame, home_team_col: str, away_team_col: str) -> pd.DataFrame():
        # Concatenate the team columns from the main df into a Series
        self.elo_df = pd.concat([df[home_team_col], df[away_team_col]])
        # Drop duplicates from the Series
        self.elo_df = self.elo_df.drop_duplicates()
        # Convert the Series to a DataFrame
        self.elo_df = self.elo_df.to_frame()
        # Name the column (which was previously unnamed)
        self.elo_df.columns = ['Team']
        # Insert another column for the score, initialized to 1200
        self.elo_df.insert(1, "Elo Score", 1200)

        return self.elo_df

    def expected_scores(self, home_team_name: str, away_team_name: str) -> [int]:
        # self.elo_df.loc['']
        pass
# Save the primary scores df for later use
primary_scores_df.to_csv('PrimaryScoresData')