import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
from numpy.random import default_rng
import pandas as pd
import make_standings_3
from make_standings_3 import Standings
from importlib import reload
import altair as alt

st.title('2021 NFL Regular Season Simulator')

df = pd.read_csv("schedules/schedule21.csv")
pr_both = pd.read_csv("data/pr_both.csv",squeeze=True,index_col="Side")
div_series = pd.read_csv("data/divisions.csv",squeeze=True,index_col=0)
teams = sorted(list(set(df.team_home)))
conf_teams = {}
for conf in ["AFC","NFC"]:
    conf_teams[conf] = [t for t in teams if div_series[t][:3]==conf]
rng = np.random.default_rng()

reps = st.sidebar.number_input('Number of simulations to run:',value=10)

st.sidebar.write("Set your power ratings:")

col1, col2 = st.sidebar.beta_columns(2)

off_pr = {}
def_pr = {}

with col1:
    for t in teams:
        off_pr[t] = st.slider(
            f'{t} offense',
            -10.0, 10.0, float(pr_both[t+"_Off"]))
    st.write("")
    off_pr["HFA"] = st.slider(
        'Home field advantage',
        0.0, 10.0, float(pr_both["HFA"]))

with col2:
    for t in teams:
        def_pr[t] = st.slider(
            f'{t} defense',
            -10.0, 10.0, float(pr_both[t+"_Def"]))
    st.write("")
    off_pr["mean_score"] = st.slider(
        'Average team score',
        0.0, 40.0, float(pr_both["mean_score"]))




def simulate_reg_season(df):
    df.loc[:,["score_home","score_away"]] = rng.normal(df[["mean_home","mean_away"]],10)
    df.loc[:,["score_home","score_away"]] = df.loc[:,["score_home","score_away"]].astype(int)
    adjust_ties(df)
    return None

def adjust_ties(df):
    tied_games = df.loc[df.score_home == df.score_away,["score_home","score_away"]].copy()
    x = rng.normal(size=len(tied_games))
    tied_games.iloc[np.where((x > -1.6) & (x < 0))[0]] += (0,3)
    tied_games.iloc[np.where(x >= 0)[0]] += (3,0)
    df.loc[tied_games.index,["score_home","score_away"]] = tied_games
    return None

for t in teams:
    pr_both[t+"_Off"] = off_pr[t]
    pr_both[t+"_Def"] = def_pr[t]
pr_both["HFA"] = off_pr["HFA"]
pr_both["mean_score"] = off_pr["mean_score"]

df["pr_home_Off"] = df.team_home.map(lambda s: pr_both[s+"_Off"])
df["pr_home_Def"] = df.team_home.map(lambda s: pr_both[s+"_Def"])
df["pr_away_Off"] = df.team_away.map(lambda s: pr_both[s+"_Off"])
df["pr_away_Def"] = df.team_away.map(lambda s: pr_both[s+"_Def"])
df["mean_home"] = df["pr_home_Off"]-df["pr_away_Def"]+pr_both["mean_score"]+pr_both["HFA"]/2
df["mean_away"] = df["pr_away_Off"]-df["pr_home_Def"]+pr_both["mean_score"]-pr_both["HFA"]/2

st.write("The simulation is based on each team's offensive and defensive power ratings.  See below for more details.")

if st.button("Run simulation"):
    placeholder = st.empty()
    placeholder2 = st.empty()
    placeholder.text(f"Running {reps} simulations of the 2021 NFL regular season")
    bar = placeholder2.progress(0.0)
    

    total_dict = {}
    win_dict = {t:{i:0 for i in range(18)} for t in teams}
    for conf in ["AFC","NFC"]:
        total_dict[conf] = {i:{t:0 for t in sorted(conf_teams[conf])} for i in range(1,8)}

    for i in range(reps):
        simulate_reg_season(df)
        stand = Standings(df)
        s = stand.standings
        s_dict = dict(zip(s.index,s.Wins))
        for t,w in s_dict.items():
            win_dict[t][w] += 1
        p = stand.playoffs
        for conf in ["AFC","NFC"]:
            for j,t in enumerate(p[conf]):
                total_dict[conf][j+1][t] += 1
        bar.progress((i+1)/reps)
    
    placeholder.text("")
    placeholder2.text("")
        

    chart_dict = {}

    for conf in ["AFC","NFC"]:

        playoff_dicts = total_dict[conf]
        
        source = pd.DataFrame([(k,t,playoff_dicts[k][t]/reps) for k in playoff_dicts.keys() for t in conf_teams[conf]],
            columns = ["Seed","Team","Proportion"])
        
        for a,b in source.groupby("Team"):
            source.loc[source["Team"] == a, "Make playoffs"] = b.Proportion.sum()

        ordering = sorted(conf_teams[conf],key=lambda t: sum([playoff_dicts[i][t] for i in playoff_dicts.keys()]),reverse=True)
        c = alt.Chart(source).mark_bar().encode(
            x=alt.X('Team',sort = ordering),
            y=alt.Y('Proportion',scale=alt.Scale(domain=[0,1])),
            tooltip = [alt.Tooltip('Seed', format=".0f"), alt.Tooltip('Proportion', format=".3f"),alt.Tooltip('Make playoffs', format=".3f")],
            color=alt.Color('Seed:N', scale=alt.Scale(scheme='tableau10'))
        ).properties(
            title=f"{conf} playoff seedings"
        )
        
        chart_dict[conf] = c
        
    playoff_charts = alt.hconcat(*chart_dict.values()).resolve_scale(
        color='independent'
    ).properties(
        title=f"Based on {reps} simulations:"
    )

    win_charts = {}
    for conf in ["AFC","NFC"]:
        source = pd.DataFrame([(w,t,win_dict[t][w]/reps) for t in conf_teams[conf] for w in range(18)],
                    columns = ["Wins","Team","Proportion"])
        
        for a,b in source.groupby("Team"):
            source.loc[source["Team"] == a,"Equal or higher"] = 1 - b.Proportion.cumsum()
        
        source["Equal or higher"] += source["Proportion"]
        
        ordering = sorted(conf_teams[conf],
            key = lambda t: sum([a*b for a,b in win_dict[t].items()])/reps, reverse = True)

        ordering_wins = list(range(18,-1,-1))

        c = alt.Chart(source).mark_bar().encode(
                y=alt.Y('Team',sort = ordering),
                x=alt.X('Proportion',scale=alt.Scale(domain=[0,1])),
                tooltip = [alt.Tooltip('Wins', format=".0f"),
                    alt.Tooltip('Proportion', format=".3f"),
                    alt.Tooltip("Equal or higher", format=".3f")],
                color=alt.Color('Wins:N', scale=alt.Scale(scheme='tableau20'),
                            sort = ordering_wins),
                order=alt.Order(
                    'Wins:N',
                    sort='descending'
                )
            ).properties(
                title=f"{conf} win totals",
            )

        overlay = pd.DataFrame({'Proportion': [0.5]})
        vline = alt.Chart(overlay).mark_rule(color='black', strokeWidth=.6).encode(x='Proportion:Q')

        win_charts[conf] = c + vline


    win_totals = alt.hconcat(*win_charts.values()).resolve_scale(
        color='independent'
    ).properties(
        title=f"Based on {reps} simulations:"
    )

    st.session_state['pc'] = playoff_charts
    st.session_state['wt'] = win_totals

if 'pc' in st.session_state:
    st.write(st.session_state['pc'])
    st.write(st.session_state['wt'])
else:
    st.write("Sample images:")
    st.image("data/pc_holder.png")
    st.image("data/wt_holder.png")

st.markdown('''
Explanations.  \n
* All computations were made in Python.  This website was made using [Streamlit](https://streamlit.io/). You can download the source code from GitHub [link missing]()
* The plots were made using [Altair](https://altair-viz.github.io/). 
In the plots from your simulation (not the placeholder images), put your mouse over the bars to get more data.
* The thin black line represents the median win total for each team, so for example, if the line
passes through the color for 9 wins, that means that the simulation suggests that 9 is the most fair number for that team's over/under win total.
* Schedule data is taken from this excellent Kaggle dataset [NFL scores and betting data](https://www.kaggle.com/tobycrabtree/nfl-scores-and-betting-data).
* The default power ratings were based on the season-long lines and totals at [Superbook](https://co.superbook.com/sports) as of July 16, 2021.
* Here is an example of what I mean by offensive and defensive power ratings:  \n  \n 
Dallas plays in Tampa in the first game of the 2021 NFL season.  \nTampa offensive 
power rating: 4.25  \nDallas defense power rating: -0.52  \nDifference: 4.25 - (-0.52) = 4.77  
We add 2.11/2 for home field advantage: 4.77 + (2.11/2) = 5.825  
The average score of a team is 23.82.  
So Tampa's expected score: 5.825 + 23.82 &#8776 29.65  \n
Dallas's expected score against Tampa is computed in the same way, except in this case we subtract (2.11/2) for home field advantage rather than adding it.  
Dallas's expected score: 1.44 - 1.21 - (2.11/2) + 23.82 &#8776 23   \n
Our default power ratings imply a spread of 6.65 and an over/under of 52.65.  As of July 16th, the actual market numbers at Pinnacle were 6.5 and 51.5.
* In the simulation, for each game, we compute the expected score for both teams as in the above example. 
We then use a normal distribution with the expected score as the mean, and with 10 as the standard deviation.
We then round to the nearest integer, and replace any negative scores with zero.
* I didn't think much about dealing with ties.  I wrote some ad hoc code that gets rid of most ties (otherwise there were as many as ten ties per season), 
with the home team slightly more likely to win in overtime than the road team.
* A team's overall power rating is the sum of its offensive power rating and its defensive power rating.  If we only care about wins and losses, and not the exact score, 
then the team's overall power rating is sufficient.  Scores of games are relevant for [NFL playoff tiebreakers](https://www.nfl.com/standings/tie-breaking-procedures).
* You can adjust the values of home field advantage and average team score at the bottom of the left-hand panel.
* No promises that my code is accurate.  (The hardest/most tedious part was implementing the tie-breaking procedures.) Please report any bugs!
''')