import math
import pandas as pd
from CleanDataframe import clean

# TODO: Convert some comments to docstrings.
df = pd.read_csv('GameData.csv')

df = clean(df)

# Create a primary scores DataFrame and drop indices with missing values.
primary_scores_df = df[['Home Team', 'Away Team', 'Date', 'Home Team Score', 'Away Team Score']].dropna()
# Sort by date
primary_scores_df = primary_scores_df.sort_values(by=['Date'])


# ELO Ranking System
class Elo:
    def __init__(self, input_df, home_team_col, away_team_col):
        """
        Number is a guess based on flat track stats data
        In the initial ELO implementation, k = 32 and exp / 400, since flat track stats uses exp / 100,
        I bumped k up by 4
        What needs to be implemented is a k function that takes in total number of games played
        """

        self.k = 128
        # Number comes from flat track stats
        self.delta = 0.06

        # Concatenate the team columns from the main df into a Series
        self.elo_ser = pd.concat([input_df[home_team_col], input_df[away_team_col]])
        # Drop duplicates from the Series
        self.elo_ser = self.elo_ser.drop_duplicates()
        # Convert the Series to a DataFrame
        self.elo_ser = self.elo_ser.to_frame()
        # Name the column (which was previously unnamed)
        self.elo_ser.columns = ['Team']
        # Insert another column for the rating, initialized to 700
        self.elo_ser.insert(1, "Elo Rating", 700)
        # The rows are unnamed, but the first column has the team names, convert the unnamed columns
        # to team names
        self.elo_ser = self.elo_ser.set_axis(self.elo_ser['Team'], axis=0)
        # Remove the team name column, thus making the type of elo_ser from a DataFrame to a Series
        self.elo_ser = self.elo_ser["Elo Rating"]

    #   We use Difference over Sum (DoS or dos) to better reflect a team's dominance in a game.
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

# TODO: Implement glicko-2 ranking algorithm
# Implementation of this document: http://www.glicko.net/glicko/glicko2.pdf
class Glicko2:
    def __init__(self, input_df, home_team_col, away_team_col):
        self.__tau = 0.75
        self.__epsilon = 0.000001

        # Create a DataFrame consisting of the team names in the same fashion as the Elo class.
        self.glicko_df = pd.concat(input_df[home_team_col], input_df[away_team_col])
        self.glicko_df = self.glicko_df.drop_duplicates()
        self.glicko_df = self.glicko_df.to_frame()
        self.glicko_df.columns = ['Team']

        # Create and populate the columns of the glicko_df
        self.glicko_df.insert(1, 'Glicko-2 Rating', 1500)
        self.glicko_df.insert(2, 'Rating Deviation', 350)
        self.glicko_df.insert(3, 'Rating Volatility', 0.06)
        self.glicko_df.insert(4, 'V', 0)
        self.glicko_df.insert(5, 'Delta', 0)

        # Rename rows to the values in the first column (which are the team names)
        self.glicko_df = self.glicko_df.set_axis(self.glicko_df['Team'], axis=0)
        # Drop the first column
        self.glicko_df = self.glicko_df.drop(columns='Team')

    # mu is the rating normalized to the glicko2 scale
    def get_mu(self, team_name):
        team_rating = self.glicko_df.loc[team_name, 'Glicko-2 Rating']
        return (team_rating - 1500) / 173.7178

    #
    def get_phi(self, team_name):
        team_rd = self.glicko_df.loc[team_name, 'Rating Deviation']
        return team_rd / 173.7178

    def get_g(self, home_team_name):
         return 1.0 / math.sqrt(1 + 3 * self.get_phi(home_team_name)**2 / math.pi**2)

    def get_e(self, home_team_name, away_team_name):
        g = self.get_g(home_team_name)
        return 1.0 / (1 + math.exp(-1.0 * g * (self.get_mu(home_team_name) - self.get_mu(away_team_name))))
    def get_v(self, home_team_name, away_team_name):
        team_v = self.glicko_df.loc[home_team_name, 'V']
        g = self.get_g(home_team_name)
        e = self.get_e(home_team_name, away_team_name)

        return (team_v + g**2 * e * (1 - e))**-1

    def get_delta(self, s, home_team_name, away_team_name):
        team_v = self.glicko_df.loc[home_team_name, 'V']
        team_delta = self.glicko_df.loc[home_team_name, 'Delta']
        g = self.get_g(home_team_name)
        e = self.get_e(home_team_name, away_team_name)

        return team_v * (team_delta + (g * (s - e)))

    def get_new_vol(self, home_team_name, away_team_name, home_team_score, away_team_score):
        sigma = self.glicko_df.loc[home_team_name, 'Rating Volatility']
        phi = self.get_phi(home_team_name)
        v = self.get_v(home_team_name, away_team_name)

        if home_team_score > away_team_score:
            s = 1
        elif home_team_score == away_team_score:
            s = 0.5
        else:
            s = 0
        delta = self.get_delta(s, home_team_name, away_team_name)

        a = math.log(sigma**2)

        A = a
        if delta**2 > phi**2 + v:
            B = math.log(delta**2 - phi**2 - v)
        else:
            k = 1
             

elo_instance = Elo(df, 'Home Team', 'Away Team')

# Need to find a vectorized solution. Until then, a for loop will have to do.
for index, match in primary_scores_df.iterrows():
    new_home_rank, new_away_rank = elo_instance.update_ranking(match)
    elo_instance.elo_ser.loc[match.loc['Home Team']] = int(new_home_rank)
    elo_instance.elo_ser.loc[match.loc['Away Team']] = int(new_away_rank)

# Save the elo df for later use
elo_instance.elo_ser.to_csv('elo_ranks.csv')
# Save the primary scores df for later use
primary_scores_df.to_csv('PrimaryScoresData.csv')
