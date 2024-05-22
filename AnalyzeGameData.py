import math
import numpy as np
import pandas as pd
from CleanDataframe import clean

# TODO: Convert some comments to docstrings.
df = pd.read_csv('GameData.csv')

df = clean(df)

# Create a primary scores DataFrame and drop indices with missing values.
primary_scores_df = df[['Home Team', 'Away Team', 'Date',
                        'Home Team Score', 'Away Team Score']].dropna()
# Sort by date
primary_scores_df = primary_scores_df.sort_values(by=['Date'])

# Save the primary scores df for later use
primary_scores_df.to_csv('PrimaryScoresData.csv')


# ELO Ranking System
class Elo:
    def __init__(self, input_df, home_team_col, away_team_col):
        """
        Number is a guess based on flat track stats data
        In the initial ELO implementation, k = 32 and exp / 400,
        since flat track stats uses exp / 100,
        I bumped k up by 4
        What needs to be implemented is a k function that takes
        in total number of games played
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
        # The rows are unnamed, but the first column has the team names,
        # convert the unnamed columns
        # to team names
        self.elo_ser = self.elo_ser.set_axis(self.elo_ser['Team'], axis=0)
        # Remove the team name column, thus making the type of elo_ser from
        # a DataFrame to a Series
        self.elo_ser = self.elo_ser["Elo Rating"]

    #   We use Difference over Sum (DoS or dos) to better reflect
    #   a team's dominance in a game.
    def expected_dos(self, match: pd.Series) -> int:
        home_team_ranking = self.elo_ser.loc[match.loc['Home Team']]
        away_team_ranking = self.elo_ser.loc[match.loc['Away Team']]

        exp = np.e ** ((away_team_ranking - home_team_ranking - self.delta)
                         / 100)
        dos = -1.0 + (2.0 / (1.0 + exp))

        return dos

    def update_ranking(self, match: pd.Series) -> [int]:
        total_match_score = (1.0 * match.loc['Home Team Score'] +
                             match.loc['Away Team Score'])
        # Expected Scores
        e_dos = self.expected_dos(match)

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


# Implementation of this document: http://www.glicko.net/glicko/glicko2.pdf
class Glicko2:

    def __init__(self, input_df, home_team_col, away_team_col):
        self.__tau = 0.75
        self.__epsilon = 0.000001

        # Create a DataFrame consisting of the team names in the same
        # fashion as the Elo class.
        self.glicko_df = pd.concat([input_df[home_team_col],
                                    input_df[away_team_col]])
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
        return 1.0 / math.sqrt(1 + 3 * self.get_phi(home_team_name) ** 2 /
                               math.pi ** 2)

    def get_e(self, home_team_name, away_team_name):
        g = self.get_g(home_team_name)
        home_mu = self.get_mu(home_team_name)
        away_mu = self.get_mu(away_team_name)

        return (1.0 / (1 + np.exp(-1.0 * g * (home_mu - away_mu))))

    def get_v(self, home_team_name, away_team_name):
        team_v = self.glicko_df.loc[home_team_name, 'V']
        g = self.get_g(home_team_name)
        e = self.get_e(home_team_name, away_team_name)

        # We have to invert team_v in order to get the summation\
        # expression on it's own. Otherwise we cannot add to it.
        # In the case team_v is zero, we set it's inverse also to zero
        if team_v == 0:
            team_v_inverse = 0
        else:
            team_v_inverse = 1.0 / team_v

        return 1.0 / (team_v_inverse + g * g * e * (1 - e))

    def get_delta(self, s, home_team_name, away_team_name):
        team_v = self.glicko_df.loc[home_team_name, 'V']
        team_delta = self.glicko_df.loc[home_team_name, 'Delta']
        g = self.get_g(home_team_name)
        e = self.get_e(home_team_name, away_team_name)

        return team_v * (team_delta + (g * (s - e)))

    # f is taken from step 5 of the pdf
    def f(self, x, delta, phi, v, a):
        num1 = math.e ** x * (delta ** 2 - phi ** 2 - v - math.e ** x)
        dem1 = 2 * (phi ** 2 + v + math.e ** x) ** 2

        num2 = x - a
        dem2 = self.__tau ** 2

        return (num1 / dem1) - (num2 / dem2)

    # Remember that the score variable is an indicator for win/loss, so we\
    # have to use a conditional.
    def get_s(self, home_team_score: int, away_team_score: int) -> float:
        if home_team_score > away_team_score:
            return 1.0
        elif home_team_score < away_team_score:
            return 0.0
        else:
            return 0.5

    def get_new_vol(self, home_team_name):
        # Initial variables
        sigma = self.glicko_df.loc[home_team_name, 'Rating Volatility']
        phi = self.get_phi(home_team_name)
        v = self.glicko_df.loc[home_team_name, 'V']

        delta = self.glicko_df.loc[home_team_name, 'Delta']

        # Step 5 part 1
        a = np.log(sigma ** 2)

        # Step 5 part 2
        A = a
        if delta ** 2 > phi ** 2 + v:
            B = math.log(delta ** 2 - phi ** 2 - v)
        else:
            k = 1
            x = a - k * self.__tau
            while self.f(x, delta, phi, v, a) < 0:
                k += 1
                x = a - k * self.__tau

            B = a - k * self.__tau

        # Step 5 part 3
        fA = self.f(A, delta, phi, v, a)
        fB = self.f(B, delta, phi, v, a)

        # Step 5 part 4
        while abs(B - A) > self.__epsilon:
            # part 4a
            C = A + (A - B) * fA / (fB - fA)
            fC = self.f(C, delta, phi, v, a)

            # part 4b
            if fC * fB <= 0:
                A = B
                fA = fB
            else:
                fA = fA / 2.0

            # part 4c
            B = C
            fB = fC

        # Step 5 part 5
        new_sigma = math.e ** (A / 2)

        return new_sigma

    # Step 6 and Step 7 part 1:
    def get_new_phi(self, home_team_name):

        phi = self.get_phi(home_team_name)
        v = self.glicko_df.loc[home_team_name, 'V']
        new_vol = self.get_new_vol(home_team_name)

        # Step 6
        inverse_phi_star_sq = 1.0 / (phi**2 + new_vol**2)
        inverse_v = 1.0 / v

        return 1.0 / math.sqrt(inverse_phi_star_sq + inverse_v)

    # Step 7 part 2
    def get_new_mu(self, home_team_name, new_phi):
        mu = self.get_mu(home_team_name)

        # Note that the summation as defined in Step 3 is the same
        # as the delta value when divided by v
        v = self.glicko_df.loc[home_team_name, 'V']
        delta = self.glicko_df.loc[home_team_name, 'Delta']
        summa = v / delta

        return mu + new_phi**2 * summa

    # Perform all steps in order and update ratings
    def update_rating(self, home_team_name, away_team_name,
                      home_team_score, away_team_score):

        # Steps 1, 2, 3 are all performed inside other steps

        # Step 3 - get new v
        home_team_v = self.get_v(home_team_name, away_team_name)
        away_team_v = self.get_v(away_team_name, home_team_name)

        # Step 3 - set new v
        self.glicko_df.loc[home_team_name, 'V'] = int(home_team_v)
        self.glicko_df.loc[away_team_name, 'V'] = int(away_team_v)

        # Step 4 - get new delta
        s = self.get_s(home_team_score, away_team_score)
        home_team_delta = self.get_delta(s, home_team_name, away_team_name)
        away_team_delta = self.get_delta(s, away_team_name, home_team_name)

        # Step 4 - set new delta
        self.glicko_df.loc[home_team_name, 'Delta'] = int(home_team_delta)
        self.glicko_df.loc[away_team_name, 'Delta'] = int(away_team_delta)

        # Step 5, 6 and 7a
        home_team_new_phi = self.get_new_phi(home_team_name)
        away_team_new_phi = self.get_new_phi(away_team_name)

        # Step 7b
        home_team_new_mu = self.get_new_mu(home_team_name, home_team_new_phi)
        away_team_new_mu = self.get_new_mu(away_team_name, away_team_new_phi)

        # Step 8 - get new ratings and rating deviations
        new_home_rank = 173.7178 * home_team_new_mu + 1500
        new_away_rank = 173.7178 * away_team_new_mu + 1500

        new_home_rd = 173.7178 * home_team_new_phi
        new_away_rd = 173.7178 * away_team_new_phi

        # Step 8 - set new ratings and rating deviations
        self.glicko_df.loc[home_team_name, 'Glicko-2 Rating'] = int(new_home_rank)
        self.glicko_df.loc[away_team_name, 'Glicko-2 Rating'] = int(new_away_rank)

        self.glicko_df.loc[home_team_name, 'Rating Deviation'] = int(new_home_rd)
        self.glicko_df.loc[away_team_name, 'Rating Deviation'] = int(new_away_rd)


elo_instance = Elo(df, 'Home Team', 'Away Team')

# Need to find a vectorized solution. Until then, a for loop will have to do.
for index, match in primary_scores_df.iterrows():
    new_home_rank, new_away_rank = elo_instance.update_ranking(match)
    elo_instance.elo_ser.loc[match.loc['Home Team']] = int(new_home_rank)
    elo_instance.elo_ser.loc[match.loc['Away Team']] = int(new_away_rank)

# Save the elo df for later use
elo_instance.elo_ser.to_csv('elo_ranks.csv')

glicko_instance = Glicko2(df, 'Home Team', 'Away Team')

for index, match in primary_scores_df.iterrows():
    home_team_name = match.loc['Home Team']
    away_team_name = match.loc['Away Team']
    home_team_score = int(match.loc['Home Team Score'])
    away_team_score = int(match.loc['Away Team Score'])

    # Using difference over sum for scores instead of raw scores
    home_dos = (home_team_score - away_team_score) / (home_team_score + away_team_score)
    away_dos = (away_team_score - home_team_score) / (home_team_score + away_team_score)

    # glicko_instance.update_rating(home_team_name, away_team_name,
    #                               home_dos, away_dos)

