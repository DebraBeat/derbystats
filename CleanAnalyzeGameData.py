import math
import pandas as pd

df = pd.read_csv('GameData.csv')

# While we've converted our data from a number of Excel sheets to a dataFrame, it still needs to be cleaned.
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
class Elo:
    def __init__(self, home_team_col, away_team_col):
        # Number is a guess based on flat track stats data
        # In the initial ELO implementation, k = 32 and exp / 400, since flat track stats uses exp / 100,
        # I bumped k up by 4
        # What needs to be implemented is a k function that takes in total number of games played
        self.k = 128
        # Number comes from flat track stats
        self.delta = 0.06

        # Concatenate the team columns from the main df into a Series
        self.elo_ser = pd.concat([df[home_team_col], df[away_team_col]])
        # Drop duplicates from the Series
        self.elo_ser = self.elo_ser.drop_duplicates()
        # Convert the Series to a DataFrame
        self.elo_ser = self.elo_ser.to_frame()
        # Name the column (which was previously unnamed)
        self.elo_ser.columns = ['Team']
        # Insert another column for the score, initialized to 700
        self.elo_ser.insert(1, "Elo Score", 700)
        self.elo_ser = self.elo_ser.set_axis(self.elo_ser['Team'], axis=0)
        self.elo_ser = self.elo_ser["Elo Score"]

    def expected_dos(self, match: pd.Series) -> int:
        home_team_ranking = self.elo_ser.loc[match.loc['Home Team']]
        away_team_ranking = self.elo_ser.loc[match.loc['Away Team']]

        exp = math.e**((away_team_ranking - home_team_ranking - self.delta) / 100)
        dos = -1.0 + (2.0 / (1.0 + exp))

        return dos

    def update_ranking(self, match: pd.Series) -> [int]:
        total_match_score = 1.0 * match.loc['Home Team Score'] + match.loc['Away Team Score']
        # Expected Scores
        e_dos =  self.expected_dos(match)

        home_team_ranking = self.elo_ser.loc[match.loc['Home Team']]
        away_team_ranking = self.elo_ser.loc[match.loc['Away Team']]

        # Actual Scores
        s_home = match.loc['Home Team Score']
        s_away = match.loc['Away Team Score']

        s_dos_home = (s_home - s_away) / total_match_score
        s_dos_away = (s_away - s_home) / total_match_score

        new_home_rank = home_team_ranking + self.k * (s_dos_home - e_dos)
        new_away_rank = away_team_ranking + self.k * (s_dos_away - e_dos)

        return [new_home_rank, new_away_rank]

class Glicko2:
    def __init__(self):
        pass

elo_instance = Elo('Home Team', 'Away Team')

# Need to find a vectorized solution. Until then, a for loop will have to do.
for index, match in primary_scores_df.iterrows():
    new_home_rank, new_away_rank = elo_instance.update_ranking(match)
    elo_instance.elo_ser.loc[match.loc['Home Team']] = int(new_home_rank)
    elo_instance.elo_ser.loc[match.loc['Away Team']] = int(new_away_rank)

# Save the elo df for later use
elo_instance.elo_ser.to_csv('elo_ranks')
# Save the primary scores df for later use
primary_scores_df.to_csv('PrimaryScoresData')
