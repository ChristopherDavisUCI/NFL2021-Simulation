import pandas as pd
import numpy as np
from numpy.random import default_rng

df = pd.read_csv("schedules/schedule2021.csv")
pr_default = pd.read_csv("data/pr_both.csv",squeeze=True,index_col="Side")
pr_custom = pd.Series()
teams = sorted(list(set(df.team_home)))
rng = default_rng()

def simulate_reg_season(pr = pr_default):
    try:
        df["pr_home_Off"] = df.team_home.map(lambda s: pr_custom[s+"_Off"])
        df["pr_home_Def"] = df.team_home.map(lambda s: pr_custom[s+"_Def"])
        df["pr_away_Off"] = df.team_away.map(lambda s: pr_custom[s+"_Off"])
        df["pr_away_Def"] = df.team_away.map(lambda s: pr_custom[s+"_Def"])
        df["mean_home"] = df["pr_home_Off"]-df["pr_away_Def"]+pr_custom["mean_score"]+pr["HFA"]/2
        df["mean_away"] = df["pr_away_Off"]-df["pr_home_Def"]+pr_custom["mean_score"]-pr["HFA"]/2
    except:
        make_pr_custom(pr)
        df["pr_home_Off"] = df.team_home.map(lambda s: pr_custom[s+"_Off"])
        df["pr_home_Def"] = df.team_home.map(lambda s: pr_custom[s+"_Def"])
        df["pr_away_Off"] = df.team_away.map(lambda s: pr_custom[s+"_Off"])
        df["pr_away_Def"] = df.team_away.map(lambda s: pr_custom[s+"_Def"])
        df["mean_home"] = df["pr_home_Off"]-df["pr_away_Def"]+pr_custom["mean_score"]+pr["HFA"]/2
        df["mean_away"] = df["pr_away_Off"]-df["pr_home_Def"]+pr_custom["mean_score"]-pr["HFA"]/2
    scores = ["score_home","score_away"]
    df.loc[:,scores] = rng.normal(df[["mean_home","mean_away"]],10)
    df.loc[:,scores] = df.loc[:,scores].astype(int)
    df[scores] = df[scores].mask(df[scores] < 0, 0)
    adjust_ties(df)
    return df

def make_pr_custom(pr):
    global pr_custom
    for s in teams:
        if (s+"_Off" in pr.keys()) and (s+"_Off" in pr.keys()):
            pr_custom[s+"_Off"] = pr[s+"_Off"]
            pr_custom[s+"_Def"] = pr[s+"_Def"]
        elif s in pr.keys():
            pr_custom[s+"_Off"] = pr[s]/2
            pr_custom[s+"_Def"] = pr[s]/2
        else:
            pr_custom[s+"_Off"] = pr_default[s+"_Off"]
            pr_custom[s+"_Def"] = pr_default[s+"_Def"]
    for x in ["mean_score","HFA"]:
        if x in pr.keys():
            pr_custom[x] = pr[x]
        else:
            pr_custom[x] = pr_default[x]
    return pr_custom

def adjust_ties(df):
    tied_games = df.loc[df.score_home == df.score_away,["score_home","score_away"]].copy()
    x = rng.normal(size=len(tied_games))
    tied_games.iloc[np.where((x > -1.6) & (x < 0))[0]] += (0,3)
    tied_games.iloc[np.where(x >= 0)[0]] += (3,0)
    df.loc[tied_games.index,["score_home","score_away"]] = tied_games
    return None