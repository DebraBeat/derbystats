import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import poisson, skellam
from CleanDataframe import clean

df = pd.read_csv('GameData.csv')
df = clean(df)

# jam_cols = []
#
# for s in list(df.columns):
#     if 'Jam' in s:
#         jam_cols.append(s)
#
# jam_df = df[jam_cols]
# home_jam_df = jam_df.iloc[:, 0:78]

score_df = df[['Home Team', 'Away Team', 'Home Team Score', 'Away Team Score']]

# We need find the valid interval and get rid of outliers in our score_df DataFrame
# Let's start with the home team
home_IQR = (score_df['Home Team Score'].quantile(0.75) -
            score_df['Home Team Score'].quantile(0.25))

home_VI = [score_df['Home Team Score'].quantile(0.25) - 1.5 * home_IQR,
           score_df['Away Team Score'].quantile(0.75) + 1.5 * home_IQR]

score_df = score_df.drop(score_df[(score_df['Home Team Score'] > home_VI[1]) |
                                  (score_df['Home Team Score'] < home_VI[0])].index)

# Away team
away_IQR = (score_df['Away Team Score'].quantile(0.75) -
            score_df['Away Team Score'].quantile(0.25))

away_VI = [score_df['Away Team Score'].quantile(0.25) - 1.5 * away_IQR,
           score_df['Away Team Score'].quantile(0.75) + 1.5 * away_IQR]

score_df = score_df.drop(score_df[(score_df['Away Team Score'] > away_VI[1]) |
                                  (score_df['Away Team Score'] < away_VI[0])].index)

# The highest score in the data set. It is set inside a method to prevent it from being overwritten
def max_score():
    return int(max(score_df['Home Team Score'].max(), score_df['Away Team Score'].max()))


# construct Poisson for each mean goals value
poisson_pred_home = [poisson.pmf(i, score_df['Home Team Score'].mean()) for i in range(max_score())]
poisson_pred_away = [poisson.pmf(i, score_df['Away Team Score'].mean()) for i in range(max_score())]
poisson_pred = np.column_stack([poisson_pred_home, poisson_pred_away])

# matplotlib version
#
# Histogram of final scores
plt.hist(score_df[['Home Team Score', 'Away Team Score']].values,
       label=['Home', 'Away'], density=True)

# Poisson distribution graphs for home and away teams
pois_home, = plt.plot([i for i in range(1, max_score()+1)], poisson_pred_home,
                      linestyle='-', marker='o', label="Home")
# pois_away, = plt.plot([i for i in range(1, max_score()+1)], poisson_pred_away,
#                       linestyle='-', marker='o', label="Away")

leg=plt.legend(loc='upper right', fontsize=13, ncol=2)
leg.set_title("Poisson           Actual        ",
              prop= {'size':'14', 'weight':'bold'})

plt.xticks([i*33 for i in range(1,11)],[i*33 for i in range(1,11)])
plt.xlabel("Final Scores",size=13)
plt.ylabel("Proportion of Matches",size=13)
plt.title("Proportion of scores per game",size=14,fontweight='bold')


plt.show()

# # seaborn version
# d = {i : poisson.pmf(i, score_df['Home Team Score'].mean()) for i in range(max_score())}
# poisson_df = pd.DataFrame(d)
# sns.histplot(data=poisson_pred,
#              x='Home Team Score')
# plt.show()

# sns.relplot(
#     data=home_jam_df.T,
#     x=[i + 1 for i in range(len(home_jam_df.columns.T))],
#     y=[10 for i in range(len(home_jam_df.columns.T))]
# )
# plt.show()
