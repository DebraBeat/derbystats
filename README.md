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
Every official inter-league game has a corresponding worksheet attached to it.
A worksheet is an excel file that contains all relevant info about the game.
We can parse these worksheets to analyze and visualize derby statistics! Luckily for us, WFTDA
maintains a list of worksheets [here.](https://drive.google.com/drive/folders/1TC1QUmpIwy9NZX9DBPUPoHjkjFbbzyYr)

In our `CreateDataFrame.py` file, we use the method
`xlsx_to_series()`, inputting our `path` to each excel spreadsheet
to convert each sheet into a pandas series.

Then we simply iterate over each spreadsheet, and append the
series to a pandas DataFrame containing data on all previous games.
 
## Rating of Derby Teams
With this in mind, it's important to rate teams to judge how strong
they are. Two popular rating algorithms are the *Elo rating algorithm*
which notoriously used in chess matches. The second
is the *glicko-2 rating algorithm*, a more complex algorithm used
in rating players in video games.

WFTDA uses it's own rating algorithm.You can read about it [here.](https://static.wftda.com/files/competition/2023-WFTDA-Rankings-Algorithm.pdf)

In my implementation of the Elo rating algorithm, 