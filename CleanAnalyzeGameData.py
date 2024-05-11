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
        # Number comes from wikipedia, could be changed later
        self.k = 32

        # Concatenate the team columns from the main df into a Series
        self.elo_ser = pd.concat([df[home_team_col], df[away_team_col]])
        # Drop duplicates from the Series
        self.elo_ser = self.elo_ser.drop_duplicates()
        # Convert the Series to a DataFrame
        self.elo_ser = self.elo_ser.to_frame()
        # Name the column (which was previously unnamed)
        self.elo_ser.columns = ['Team']
        # Insert another column for the score, initialized to 1200
        self.elo_ser.insert(1, "Elo Score", 1200)
        self.elo_ser = self.elo_ser.set_axis(self.elo_ser['Team'], axis=0)
        self.elo_ser = self.elo_ser["Elo Score"]

    def expected_scores(self, match: pd.Series) -> [int]:
        home_team_ranking = self.elo_ser.loc[match.loc['Home Team']]
        away_team_ranking = self.elo_ser.loc[match.loc['Away Team']]

        total_match_score = 1.0 * match.loc['Home Team Score'] + match.loc['Away Team Score']

        # Shorthand to make the expectation formulae below more concise
        q_home = 10.0**(home_team_ranking / 400.0)
        q_away = 10.0**(away_team_ranking / 400.0)

        # Without multiplying by total_match_score, our e_home and e_away are our probabilities of
        # each team winning. Our expectation is (by definition), the value of the score multiplied
        # by its probability.
        e_home = total_match_score * q_home / (q_home + q_away)
        e_away = total_match_score * q_away / (q_home + q_away)

        return [e_home, e_away]

    def update_ranking(self, match: pd.Series) -> [int]:
        total_match_score = 1.0 * match.loc['Home Team Score'] + match.loc['Away Team Score']
        # Expected Scores
        e_home, e_away = self.expected_scores(match)

        home_team_ranking = self.elo_ser.loc[match.loc['Home Team']]
        away_team_ranking = self.elo_ser.loc[match.loc['Away Team']]

        # Actual Scores
        s_home = match.loc['Home Team Score']
        s_away = match.loc['Away Team Score']

        dos = ()

        new_home_rank = home_team_ranking + self.k * (s_home - e_home)
        new_away_rank = away_team_ranking + self.k * (s_away - e_away)

        return [new_home_rank, new_away_rank]


# Save the primary scores df for later use
primary_scores_df.to_csv('PrimaryScoresData')
