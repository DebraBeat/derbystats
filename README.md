# Analysis and Visualization of Women's Flat Track Derby Game Data

## A crash course in Roller Derby
Before we get started in analysis and visualization of roller
derby, we should remind ourselves of what a Roller Derby game is!

From [WFTDA](https://rules.wftda.com/summary.html), the biggest roller derby league in the world: 
> The game of Flat Track Roller Derby is played on a flat, oval track.
> Play is broken up into two 30-minute periods, and within those periods, into units of play called “Jams,” which last up to two minutes. There are 30 seconds between each Jam. During a Jam, each team fields up to five Skaters. Four of these Skaters are called “Blockers” (together, the Blockers are called the “Pack”), and one is called a “Jammer.” The Jammer wears a helmet cover with a star on it.
The two Jammers start each Jam behind the Pack, and score a point for every opposing Blocker they lap, each lap. Because they start behind the Pack, they must get through the Pack, then all the way around the track to be eligible to score points on opposing Blockers.
Roller derby is a full-contact sport; however, Skaters cannot use their heads, elbows, forearms, hands, knees, lower legs, or feet to make contact to opponents. Skaters cannot make contact to opponents’ heads, backs, knees, lower legs, or feet.
Play that is unsafe or illegal may result in a Skater being assessed a penalty, which is served by sitting in the Penalty Box for 30 seconds of Jam time.
The team with the most points at the end of the game wins.

## Getting Derby Data
Every official inter-league match has a corresponding worksheet attached to it.
A worksheet is an excel file that contains all relevant info about the match.
We can parse these worksheets to analyze and visualize derby statistics! Luckily for us, WFTDA
maintains a list of worksheets [here.](https://drive.google.com/drive/folders/1TC1QUmpIwy9NZX9DBPUPoHjkjFbbzyYr)

In our `CreateDataFrame.py` file, we use the method
`xlsx_to_series()`, inputting our `path` to each excel spreadsheet
to convert each sheet into a pandas series.

Then we simply iterate over each spreadsheet, and append the
series to a pandas DataFrame containing data on all previous matches.

However this DataFrame is messy. Due to the way the data was
variously put into the worksheets, our pandas columns are not
type correct, and our indices are weird as well. We use the `clean(df)`
method from `CleanDataFrame.py` to make our
Dataframe workable. 
 
## Rating of Derby Teams
With this in mind, it's important to rate teams to judge how strong
they are. Two popular rating algorithms are the *Elo rating algorithm*
which notoriously used in chess matches. The second
is the *glicko-2 rating algorithm*, a more complex algorithm used
in rating players in video games.

WFTDA uses it's own rating algorithm.You can read about it [here.](https://static.wftda.com/files/competition/2023-WFTDA-Rankings-Algorithm.pdf)

### Elo implementation
In `AnalyzeGameData.py`, we create a Elo class to contain all relevant
methods and variables which will only be instantiated
once. Most notably, this class contains `elo_df`, a DataFrame
containing the Elo rating of each team.

In my implementation of the Elo rating algorithm, I use the
difference of scores over sum of scores instead of the raw scores, as
it better reflects which team was in control of the match.

We initialize each team to begin with a 700 rating, and go through
our match DataFrame and update `elo_df` accordingly.

In `VisualizeData.py`, we produce a graph of the distribution of
Elo ratings. Note that it roughly follows a normal distribution as
intended by the Elo algorithm.

![Elo rankings distribution.png](Elo%20rankings%20distribution.png)

### glicko-2 implementation
This implementation is still under works. Currently, the volatility
of each team's ratings makes the rating diverge (go to extreme values).
A code review is needed.

## Home vs Away teams
### Visualization
Oftentimes, an away team will want to play a higher rated
home team. Even though they will most likely lose, their
rating will increase if they score more points than is expected
by WFTDAs algorithm (similar the Elo algorithm).

Because of this, home teams often have an advantage. Let's
visualize what that advantage looks like.

![Home vs Away Scores freq.png](Home%20vs%20Away%20Scores%20freq.png)

We can see from the above that most games are pretty close,
however with the following graphs we can see the home team
has a clear advantage.

![Home vs Away JointGrid.png](Home%20vs%20Away%20JointGrid.png)
![Home vs Away Scores.png](Home%20vs%20Away%20Scores.png)

We can see from the first graph above that although the
Inner Quartile range is roughly the same for Home vs Away teams
the home team has a higher first quartile, mean, and third quartile.

From the second graph above, the distinction is even clearer.
The away team is less likely to score above 125 points.

![Poisson Distributions.png](Poisson%20Distributions.png)

In this graph we can see the poisson distribution using the mean
number of scores by home and away teams. Notice how the peaks
are the same as the KDEs, but the distribution quickly drops off to `0`.

![Skellam dist Home vs Away.png](Skellam%20dist%20Home%20vs%20Away.png)

### Analysis - Kernel Density Estimation
While visualization is a powerful tool in helping us
understand data, it is useful to help quantify it.

For fun, we can use a kernel density estimation to simulate matches
and also compare it to real world data. Although the bandwidth
is set at `0.5`, we can write code to correctly fit our KDE by
minimizing the difference between KDE outcomes and real world data.

Also note that while KDEs are fun and useful for smoothing
it is ultimately pointless. Our KDEs can estimate our statistics,
which means that simply drawing a uniform random number between
`[0,1]` and comparing it to the real world probability of winning
is a more effective way of running a simulation.

From both KDE simulations and the statistics, we see that the
home team has a winning percent of `61.37%`!

### Analysis - Poisson and Skellam PMF / CDF
In `VisualizeData.py` we create two lists of values for home
and away teams using the poisson distribution and the mean score
for each team. We can clearly see from our visualization that
the home team has an advantage. We could use the `sympy` library
to find the max value of each distribution as well.

Additionally, we create a skellam distribution to find the win/loss
probability. 

## Jams

### Visualization
![Home vs Away Jams LinePlot.png](Home%20vs%20Away%20Jams%20LinePlot.png)

From the above we can see the cumulative jam score, as well as
the first and third quartiles of the cumulative score. Not the
discrepancy around Jams 32 and 68.

![Number of Jams per Jam Number.png](Number%20of%20Jams%20per%20Jam%20Number.png)
This discrepancy can be attributed to the fact that not many jams
take place by Jam 31 and 67, as the clock usual runs out.
Additionally, we can see a discrepancy in reporting, with some recorders
starting jams at zero, and others starting jams at 1.

### Analysis
In `AnalyzeGameData.py` we use the frequency of each jam to show that
the minimum is over 500 times less than the mean! At a minimum of `2`
from over 750 games analyzed, the probability of a jam going past 32
in either half is less than `0.1%`!

We can see the highest change is at jam 72, where it is a whopping 20 points!
The high deviation in the later jams indicate two things: Firstly, there are not
a lot of jams played in those high numbers. Secondly, teams are really fighting
to survive!

## Conclusion
We can conclude that the home team has a definite advantage.
While derby is very fun to watch, it should be enjoyed for the
love of the game, not for competitive reasons.

We can also conclude that players are willing to go "all out"
towards the match, which makes it very fun to watch!