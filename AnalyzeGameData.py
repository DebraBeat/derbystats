import math
import numpy as np
import pandas as pd
from CleanDataframe import clean
from sklearn.neighbors import KernelDensity
from scipy.stats import poisson, skellam

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

        self.k = 32 # Put back to orginal elo value
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

        # New implementation taken from Ryan Kirkman glicko2 implementation:
        # https://github.com/ryankirkman/pyglicko2/blob/master/glicko2.py

        a = math.log(math.pow(sigma, 2))
        x0 = a
        x1 = 0

        while x0 != x1:
            # New iteration, so x(i) becomes x(i-1)
            x0 = x1
            d = math.pow(self.get_mu(home_team_name), 2) + v + math.exp(x0)
            h1 = -(x0 - a) / math.pow(self.__tau, 2) - 0.5 * math.exp(x0) \
            / d + 0.5 * math.exp(x0) * math.pow(delta / d, 2)
            h2 = -1 / math.pow(self.__tau, 2) - 0.5 * math.exp(x0) * \
            (math.pow(self.get_mu(home_team_name), 2) + v) \
            / math.pow(d, 2) + 0.5 * math.pow(delta, 2) * math.exp(x0) \
            * (math.pow(self.get_mu(home_team_name), 2) + v - math.exp(x0)) / math.pow(d, 3)
            x1 = x0 - (h1 / h2)

        return math.exp(x1 / 2)


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


        # print(f'home team phi: {home_team_new_phi}')
        # print(f'away team name: {away_team_name}, new rank: {new_away_rank}')


        new_home_rd = 173.7178 * home_team_new_phi
        new_away_rd = 173.7178 * away_team_new_phi

        # Step 8 - set new ratings and rating deviations
        self.glicko_df.loc[home_team_name, 'Glicko-2 Rating'] = int(new_home_rank)
        self.glicko_df.loc[away_team_name, 'Glicko-2 Rating'] = int(new_away_rank)

        self.glicko_df.loc[home_team_name, 'Rating Deviation'] = int(new_home_rd)
        self.glicko_df.loc[away_team_name, 'Rating Deviation'] = int(new_away_rd)

        self.glicko_df.loc[home_team_name, 'Rating Volatility'] = int(home_team_new_phi)
        self.glicko_df.loc[away_team_name, 'Rating Volatility'] = int(away_team_new_phi)


elo_instance = Elo(df, 'Home Team', 'Away Team')

# Need to find a vectorized solution. Until then, a for loop will have to do.
for index, match in primary_scores_df.iterrows():
    new_home_rank, new_away_rank = elo_instance.update_ranking(match)
    elo_instance.elo_ser.loc[match.loc['Home Team']] = int(new_home_rank)
    elo_instance.elo_ser.loc[match.loc['Away Team']] = int(new_away_rank)

# Save the elo df for later use
elo_instance.elo_ser.to_csv('elo_ranks.csv')

glicko_instance = Glicko2(df, 'Home Team', 'Away Team')


# TODO: Fix glicko implementation
# i = 0
# for index, match in primary_scores_df.iterrows():
#     home_team_name = match.loc['Home Team']
#     away_team_name = match.loc['Away Team']
#     home_team_score = int(match.loc['Home Team Score'])
#     away_team_score = int(match.loc['Away Team Score'])
#
#     i += 1
#     if i == 39:
#         break
#     # Using difference over sum for scores instead of raw scores
#     home_dos = (home_team_score - away_team_score) / (home_team_score + away_team_score)
#     away_dos = (away_team_score - home_team_score) / (home_team_score + away_team_score)
#
#     glicko_instance.update_rating(home_team_name, away_team_name,
#                                   home_dos, away_dos)

# Data used for Kernel Density Approximation
X_home = primary_scores_df['Home Team Score'].to_frame()
X_away = primary_scores_df['Away Team Score'].to_frame()

# Kernel Density Approximations for home and away teams
# TODO: properly fit bandwidth
kde_home = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X_home)
kde_away = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X_away)

# Putting them into a series
# Note that the KDEs are two dimensional numpy arrays, so we need to unravel them
samples = pd.DataFrame({'Home': kde_home.sample(10000000, 0).ravel(order='C'),
                        'Away': kde_away.sample(10000000, 0).ravel(order='C')})

# Scores are integers, so we round every score down
samples['Home'] = samples['Home'].astype('int')
samples['Away'] = samples['Away'].astype('int')

# The length of the DataFrame where the home team score is greater than the away team
# is the number of wins
kde_home_wins = samples[samples['Home'] > samples['Away']].shape[0]

# KDE winning percent
kde_home_win_per = kde_home_wins / samples.shape[0]

# Real life data
real_home_wins = primary_scores_df[primary_scores_df['Home Team Score'] >
                                   primary_scores_df['Away Team Score']].shape[0]

# Real life winning percent
real_home_win_per = real_home_wins / primary_scores_df.shape[0]

print(f'KDE winning percent: {kde_home_win_per * 100:.2f}')
print(f'Real life winning percent: {real_home_win_per * 100:.2f}')

def max_score():
    return int(max(primary_scores_df['Home Team Score'].max(), primary_scores_df['Away Team Score'].max()))

# construct Poisson for each mean goals value
# TODO: Use sympy to find the max of each
poisson_pred_home = [poisson.pmf(i,
                                 primary_scores_df['Home Team Score'].mean()) for i in range(max_score())]
poisson_pred_away = [poisson.pmf(i,
                                 primary_scores_df['Away Team Score'].mean()) for i in range(max_score())]

# Probability of a draw
draw_prob = skellam.pmf(0.0, primary_scores_df['Home Team Score'].mean(), primary_scores_df['Away Team Score'].mean())
print(f'Probability of a draw: {draw_prob}')

# Probability of a loss
loss_prob = skellam.cdf(0.0, primary_scores_df['Home Team Score'].mean(), primary_scores_df['Away Team Score'].mean())
print(f'Probability of a loss: {loss_prob}')

# Probability of a win = 1 - P(loss)
print(f'Probability of a win: {1 - loss_prob}')

# Jam Analysis
jam_df = pd.read_csv('jam_df.csv')
jam_freq = [jam_df[jam_df['Jam Number'] == i].shape[0] for i in range(1,77)]
jam_freq = pd.Series(jam_freq)

print(f"The mean number of jams is {jam_freq.mean / jam_freq.min} less than the minimum")

# Jam Std
jam_std = [jam_df[jam_df["Jam Number"] == i]["Jam Score"].std() for i in range(1,77)]
jam_std_change = [jam_std[i] - jam_std[i-1] for i in range(1,76)]

print(f"Jam deviation at jams 30-35: {jam_std[29:34]}")
print(f'The highest change is at Jam 72 where it is {max(jam_std_change)}')