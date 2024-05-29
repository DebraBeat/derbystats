import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import poisson, skellam
from CleanDataframe import clean

df = pd.read_csv('GameData.csv')
df = clean(df)

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

# 112 is the third quantile
d = {'Winning diff': [i for i in range(-112, 113)],
     'PMF': [skellam.pmf(i, score_df['Home Team Score'].mean(), score_df['Away Team Score'].mean())
             for i in range(-112, 113)]}

skellam_df = pd.DataFrame(d)
p = sns.lineplot(data=skellam_df, x='Winning diff', y='PMF')
p.set_title('Chance of Home Team Winning by certain number of points')
plt.xlabel("Score Difference (Home - Away)")
plt.ylabel("Probability")
# plt.savefig("Skellam dist Home vs Away")


# matplotlib version
# Poisson distribution graphs for home and away teams
pois_home, = plt.plot([i for i in range(1, max_score() + 1)], poisson_pred_home,
                      linestyle='-', label="Home")
pois_away, = plt.plot([i for i in range(1, max_score() + 1)], poisson_pred_away,
                      linestyle='-', label="Away")

leg = plt.legend(loc='upper right', fontsize=13, ncol=2)
leg.set_title("Poisson",
              prop={'size': '10', 'weight': 'bold'})

plt.xticks([i * 33 for i in range(1, 11)], [i * 33 for i in range(1, 11)])
plt.xlabel("Final Scores", size=13)
plt.ylabel("Proportion of Matches", size=13)
plt.title("Proportion of scores per game", size=14, fontweight='bold')

# plt.savefig('Poisson Distributions')
# plt.show()

# seaborn version
sns.histplot(data=score_df,
             x='Home Team Score',
             y='Away Team Score',
             cbar=True)

# plt.show()
# plt.savefig("Home vs Away Scores freq.png")
# plt.close()
plt.close('all')

p = sns.histplot(data=score_df,
                 multiple='dodge',
                 shrink=0.6,
                 kde=True,
                 stat='density')
p.set(xlabel='Points')
p.set_title('Score PMF')
# plt.show()
# plt.savefig("Home vs Away Scores.png")
# plt.close()

g = sns.JointGrid(data=score_df,
                  x="Home Team Score",
                  y="Away Team Score",
                  )
g.plot_joint(sns.histplot)
g.plot_marginals(sns.boxplot)

# plt.savefig("Home vs Away JointGrid.png")
plt.close('all')

elo_df = pd.read_csv("elo_ranks.csv")

p = sns.histplot(data=elo_df)
p.set(xlabel='Elo Rating')
p.set_title('Elo Rating Distribution')
# plt.show()
# plt.savefig("Elo rankings distribution")
plt.close('all')

# Get a list of Jam Columns
jam_cols = [f'Home Jam {i} Cumulative Score' for i in range(1, 77)] \
           + [f'Away Jam {i} Cumulative Score' for i in range(1, 77)]

# Create a long form DataFrame with the jam scores
jam_df = pd.concat([df[jam] for jam in jam_cols], keys=[jam for jam in jam_cols])
# Convert to frame and create a Jam Column, which is a tuple with the jam number and match number
jam_df = jam_df.to_frame()
jam_df['Jam'] = jam_df.index
# Break the two into two seperate columns
jam_df[['Jam', 'Game Number']] = pd.DataFrame(jam_df['Jam'].to_list(), index=jam_df.index)
# Reset the index
jam_df = jam_df.reset_index()
# Rename Jam Score column
jam_df = jam_df.rename(columns={0: 'Jam Score'})
# Drop missing values
jam_df = jam_df.dropna()
# Create a team type column to use a hue in seaborn
jam_df['Team Type'] = jam_df['Jam'].str.slice(0, 4)
# Create a jam number column to use in seaborn
jam_df['Jam Number'] = jam_df['Jam'].str.slice(9, 11)
# Convert it to ints
jam_df['Jam Number'] = jam_df['Jam Number'].astype('int')

# We need find the valid interval and get rid of outliers in our jam_df DataFrame
for jam_number in range(1, 77):
    q1 = jam_df[jam_df['Jam Number'] == jam_number]['Jam Score'].quantile(0.25)
    q3 = jam_df[jam_df['Jam Number'] == jam_number]['Jam Score'].quantile(0.75)
    iqr = q3 - q1  # Inner Quartile Range
    vi = [q1 - 1.5 * iqr, q3 + 1.5 * iqr]  # Valid interval

    outliers = jam_df[(jam_df['Jam Number'] == jam_number) &
                      ((jam_df['Jam Score'] > vi[1]) | (jam_df['Jam Score'] < vi[0]))]

    jam_df = jam_df.drop(outliers.index)

# sns.lineplot(data=jam_df, x="Jam Number", y="Jam Score", hue="Team Type")
# plt.savefig("Home vs Away Jams LinePlot.png")
# plt.show()

jam_df.to_csv("jam_df.csv")
