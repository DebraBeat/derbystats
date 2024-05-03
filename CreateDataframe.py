import pandas as pd
import os

directory = r"C:\Users\zeeri\PycharmProjects\derbystats\games"

def xlsx_to_series(path):
    df = pd.read_excel(path,
                       sheet_name="Score",
                       usecols="A,B,H:U,AA:AK")
    df = df.drop([0, 41, 42, 43], axis=0)

    date = df.columns[5]

    home = df.columns[0]
    # Unused, but gets the scores for each individual jam for the home team
    # home_jam_scores = df.iloc[1:, 11]
    home_game_total = df.iloc[1:, 12]
    home_halftime_score = df.iloc[39, 12]
    try:
        home_final_score = df.iloc[78, 12]
    except IndexError:
        print("Index Error occurred, trying above cell. (77 instead of 78)")
        home_final_score = df.iloc[77, 12]
    except Exception:
        print("Still got an error, marking score as -1 to be dropped later")
        home_final_score = -1

    away = df.columns[14]
    # Unused, but gets the scores for each individual jam for the away team
    # away_jam_scores = df.iloc[1:, 25]
    away_game_total = df.iloc[1:, 26]
    away_halftime_score = df.iloc[39, 26]
    try:
        away_final_score = df.iloc[78, 26]
    except IndexError:
        print("Index Error occurred, trying above cell. (77 instead of 78)")
        away_final_score = df.iloc[77, 26]
    except Exception:
        print("Still got an error, marking score as -1 to be dropped later")
        away_final_score = -1

    home_jam_keys = [f'Home Jam {i} Cumulative Score' for i in range(1, 79)]
    away_jam_keys = [f'Away Jam {i} Cumulative Score' for i in range(1, 79)]

    d = {'Home Team': home,
         'Away Team': away,
         'Date': date,
         'Home Team Score': home_final_score,
         'Away Team Score': away_final_score,
         'Home Team Halftime Score': home_halftime_score,
         'Away Team Halftime Score': away_halftime_score
         }

    for key, row_value in zip(home_jam_keys, home_game_total.items()):
        d[key] = row_value[1]

    for key, row_value in zip(away_jam_keys, away_game_total.items()):
        d[key] = row_value[1]

    ser = pd.Series(data=d, index=d.keys())

    return ser

df = pd.DataFrame()
i = 1
total_games = len(os.listdir(directory))

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)

    ser = xlsx_to_series(f)

    home, away = ser.loc['Home Team'], ser.loc['Away Team']
    print(f'Now working on {home} vs {away}')
    print(f'Number {i} out of {total_games}')
    i += 1

    ser = ser.to_frame()
    ser = ser.transpose()
    df = pd.concat([ser, df])

df.to_csv('GameData.csv')
# Slow runtime, but easy to check that the data is organized correctly
# df.to_excel('test.xlsx')