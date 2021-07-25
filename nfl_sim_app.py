import streamlit as st
from numpy.random import default_rng
import pandas as pd
from sim_season import simulate_reg_season, make_pr_custom
from make_standings import Standings
from make_charts import make_playoff_charts, make_win_charts
from itertools import permutations
import time
import base64

st.set_page_config(layout="wide")

st.title('2021 NFL Regular Season Simulator')

pr_both = pd.read_csv("data/pr_both.csv",squeeze=True,index_col="Side")
div_series = pd.read_csv("data/divisions.csv",squeeze=True,index_col=0)
teams = div_series.index
conf_teams = {}
for conf in ["AFC","NFC"]:
    conf_teams[conf] = [t for t in teams if div_series[t][:3]==conf]
rng = default_rng()

div_dict = {}

s = div_series
for a,b in s.groupby(s):
    div_dict[a] = list(b.index)

def reps_changed():
    st.session_state["rc"] = True

def prob_to_odds(p):
    if p < .000001:
        return "NA"
    if p > .999999:
        return "NA"
    if p > 0.5:
        x = 100*p/(p-1)
        return f"{x:.0f}"
    elif p <= 0.5:
        x = 100*(1-p)/p
        return f"+{x:.0f}"

def get_table_download_link(df,filename,text="Download csv file"):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

max_reps = 10**4

reps = st.sidebar.number_input('Number of simulations to run:',value=10, min_value = 1, max_value = max_reps, help = f'''The maximum allowed value is {max_reps}.''',on_change=reps_changed)

format_dict = {"Combined": "Combined (one rating per team)",
            "Separate":"Separate offensive/defensive ratings"}

pr_select = st.sidebar.selectbox(
        "Which type of power ratings do you want to use?",
        options = ["Combined","Separate"],
        format_func = lambda s: format_dict[s],
        index = 0,
        on_change=reps_changed
    )

info_col0, info_col1 = st.sidebar.beta_columns((3,1))

with st.sidebar:

    st.write("Set your power ratings.")

    info_expander = st.beta_expander("Expand for more details.", expanded=False)

with info_expander:
    if pr_select == "Combined":
        st.write("Each team's power rating represents the point spread against an average team on a neutral field.")
        st.subheader('Example with default ratings:')
        st.code('''
Dallas plays in Tampa Week 1
Tampa power rating: 5.46
Dallas power rating: 0.92
5.46 - 0.92 = 4.54
We add 2.11 for home field:
4.54 + 2.11 = 6.65''')
        st.write('''That number 6.65 represents our predicted spread for Dallas at Tampa.  The outcomes in our simulation are random, but the average outcome will
be a Tampa victory by 6.65 points.''')
        st.write("The team power rankings, 1-32, are shown at the bottom of the main panel on the right.")
    elif pr_select == "Separate":
        st.write('''We use 23.82 as the average score of an average team in an NFL game.
How much more or less a team scores than average is determined by the power ratings.''')
        st.subheader('Example with default raings:')
        st.code('''
Dallas plays in Tampa Week 1
Tampa offensive PR: 4.25
Dallas defensive PR: -0.52
4.25 - (-0.52) = 4.77
We add 2.11/2 for home field:
4.77 + (2.11/2) = 5.825''')
        st.write('''That number 5.825 represents how many points more than 23.82 we expect Tampa to score.  The outcomes in our simulation are random,
but the average outcome will be that Tampa scores 23.82 + 5.825 points.''')
        st.write('''A team's overall power rating represents the point spread against an average team on a neutral field.
A team's overall power rating is the sum of its offensive and defensive power ratings.''')
        st.write("The current team power rankings, 1-32, are shown at the bottom of the main panel on the right.")

pr = {}

if pr_select == "Separate":
    with st.sidebar:
        col1, col2 = st.beta_columns(2)

        with col1:
            for t in teams:
                pr[t+"_Off"] = st.slider(
                    f'{t} offense',
                    -10.0, 10.0, float(pr_both[t+"_Off"]))

        with col2:
            for t in teams:
                pr[t+"_Def"] = st.slider(
                    f'{t} defense',
                    -10.0, 10.0, float(pr_both[t+"_Def"]))

        st.subheader("Extra parameters:")
        pr["HFA"] = st.sidebar.slider(
            'Home field advantage',
            0.0, 10.0, float(pr_both["HFA"]))
        pr["mean_score"] = st.sidebar.slider(
                'Average team score',
                0.0, 40.0, float(pr_both["mean_score"]))
        pr_complete = pr
elif pr_select == "Combined":
    with st.sidebar:
        for t in teams:
            pr[t] = st.slider(
                f'{t} power rating',
                -15.0, 15.0, float(pr_both[t+"_Off"]+pr_both[t+"_Def"]))
        st.subheader("Extra parameters:")
        pr["HFA"] = st.slider(
            'Home field advantage',
            0.0, 10.0, float(pr_both["HFA"]))
        pr["mean_score"] = st.slider(
            'Average team score',
            0.0, 40.0, float(pr_both["mean_score"]))
        pr_complete = make_pr_custom(pr)


with st.sidebar:
    series_download = pd.Series(pr,name="PR")
    series_download.index.name = "Side"
    st.write("")
    st.markdown(get_table_download_link(series_download,"pr.csv","Download your power ratings as a csv file"), unsafe_allow_html=True)



df_pr = pd.DataFrame({"Overall": {t:pr_complete[t+"_Off"] + pr_complete[t+"_Def"] for t in teams},
        "Offensive":{t:pr_complete[t+"_Off"] for t in teams}, "Defensive":{t:pr_complete[t+"_Def"] for t in teams}}
        )


st.markdown('''Based on your power ratings, we use simulations of the 2021 regular season to estimate answers to questions like:\n* How likely is Cleveland to get the no. 1 seed?  To win its division?
To make the playoffs?\n* How likely are the Steelers to win exactly 11 games?  To win 11 or more games?\n* What is the most likely exact finish 1-4 of teams in the AFC North?''')

st.write('''In each simulation, a random outcome is generated for all 272 regular season games.  The outcomes are based on the power ratings (see below for more details).
You should customize these power ratings on the left.  You can also adjust the offensive and defensive power ratings separately.
The separate ratings are important for playoff tie-breakers, but for wins and losses, the overall power ratings are sufficient.
Click the button below to run the simulation.''')

button_cols1, button_cols2 = st.beta_columns((1,5))

with button_cols1:
    sim_button = st.button("Run simulations")

with button_cols2:
    time_holder = st.empty()

if sim_button or ("rc" in st.session_state):
    try:
        del st.session_state["rc"]
    except:
        pass
    st.header("Simulation results")
    placeholder0 = st.empty()
    placeholder1 = st.empty()
    placeholder0.text(f"Running {reps} simulations of the 2021 NFL regular season")
    bar = placeholder1.progress(0.0)
    st.write("")

    playoff_dict = {}
    
    for conf in ["AFC","NFC"]:
        playoff_dict[conf] = {i:{t:0 for t in sorted(conf_teams[conf])} for i in range(1,8)}

    win_dict = {t:{i:0 for i in range(18)} for t in teams}

    rank_dict = {div:{} for div in div_dict.keys()}

    for div in rank_dict.keys():
        for team_sort in permutations(div_dict[div]):
            rank_dict[div][team_sort] = 0

    start = time.time()

    for i in range(reps):
        df = simulate_reg_season(pr)
        stand = Standings(df)

        p = stand.playoffs
        for conf in ["AFC","NFC"]:
            for j,t in enumerate(p[conf]):
                playoff_dict[conf][j+1][t] += 1
        for t in teams:
            w = stand.standings.loc[t,"Wins"]
            win_dict[t][w] += 1
        
        for d in rank_dict.keys():
            rank_dict[d][tuple(stand.div_ranks[d])] += 1
        
        bar.progress((i+1)/reps)

    for d in rank_dict.keys():
        rank_dict[d] = {i:j/reps for i,j in rank_dict[d].items()}

    st.session_state["rd"] = rank_dict

    end = time.time()
    
    time_holder.write(f"{reps} simulations of the 2021 NFL regular season took {end - start:.1f} seconds.")

    playoff_charts = make_playoff_charts(playoff_dict)

    win_charts = make_win_charts(win_dict)

    st.session_state['pc'] = playoff_charts
    st.session_state['wt'] = win_charts


def make_ranking(df,col):
    return (-df[col]).rank()

def make_sample():
    st.header("Sample images:")
    if "pc" not in st.session_state:
        st.write('(To replace the sample images with real images, press the "Run simulations" button above.)')
    c_image, c_text = st.beta_columns(2)
    with c_image:
        st.image("data/pc_holder.png")
    with c_text:
        st.subheader("How to interpret the playoff seeding image.")
        st.markdown('''The AFC playoff seeding image shows the probability of different teams getting different playoff seeds, according to our simulations.\n\nFor example:
* Kansas City has over a 90% chance of making the playoffs, while Houston has less than a 5% chance.  This corresponds to the total heights of the bars in the chart.
* Kansas City is much more likely to get a 1 seed (dark blue bar) than Indianapolis (about 30% vs 5%).
* But Kansas City is much less likely to get a 4 seed (light blue bar) than Indianoplis (about 10% vs 25%). 
* Kansas City has over a 70% chance of getting a top 4 seed (dark blue + orange + red + light blue), while Indianapolis has about a 56% chance of getting a top 4 seed.
(Getting a top 4 seed is the same as winning the division.)\n\n
For more precise numbers, place your mouse over a bar in one of the real images (not the sample images).
The displayed text in our sample image shows that, according to our simulations:
* Cleveland has a 5% chance of getting a 4 seed;
* Cleveland has a 36.2% chance of winning its division;
* Cleveland has a 69.4% chance of making the playoffs.    
* The odds corresponding to these probabilities are +1900, +176, -227, respectively.''')
    c_image, c_text = st.beta_columns(2)
    with c_image:
        st.image("data/wt_holder.png")
    with c_text:
        st.subheader("How to interpret the win total image.")
        st.markdown('''The AFC win total image shows the probability, according to our simulations, of different teams having a specific number of wins at the end of the 2021 regular season.
The thin black line represents the median win total for each team.\n\nFor example:
* The thin black vertical line passes through the median win total for each team.  The median win total for Kansas City is 12, the median win total for Cleveland is 10, while the median win total for Houston is 5.
* It looks like Baltimore has about a 48% chance of winning more than 10 games (the area to the left of the yellow bar), 
about a 32% chance of winning less than 10 games (the area to the right of the yellow bar), and about a 20% chance of winning exactly 10 games (the yellow bar).
* Pittsburgh has a 10.2% chance of winning exactly 11 games, and a 17.6% chance of winning at least 11 games.''')

if 'pc' in st.session_state:
    try:
        placeholder0.write(st.session_state['pc'])
        placeholder1.write(st.session_state['wt'])
    except:
        st.header("Simulation results")
        st.write(st.session_state['pc'])
        st.write(st.session_state['wt'])
else:
    make_sample()
    

df_rankings = pd.DataFrame({col:make_ranking(df_pr,col) for col in df_pr.columns})

if pr_select == "Separate":
    rankings = st.beta_expander("Expand to see the power rankings based on the current values. (Click a column header to sort.)", expanded=False)
    with rankings:
        st.dataframe(df_rankings.sort_values("Overall"),height=500)
elif pr_select == "Combined":
    rankings = st.beta_expander("Expand to see the power rankings based on the current values.", expanded=False)
    with rankings:
        st.dataframe(df_rankings[["Overall"]].sort_values("Overall").transpose())

expand_div = st.beta_expander("Expand to see exact division outcomes.", expanded=False)
with expand_div:
    if "rd" in st.session_state:
        show_div = st.selectbox(label="Display the most likely outcomes for this division:",options = div_dict.keys())
        rank_dict = st.session_state["rd"]
        sorted_order = sorted(rank_dict[show_div].keys(),key=lambda x: rank_dict[show_div][x],reverse=True)
        st.write(f"Here are all the exact outcomes for the {show_div} which occurred at least 1% of the time during the simulation:")
        for i in [x for x in sorted_order if rank_dict[show_div][x] >= .01]:
            st.write('  '.join([f"{n+1}.&nbsp{i[n]}&nbsp&nbsp" for n in range(4)])+f"&nbsp&nbsp Proportion: {rank_dict[show_div][i]:.3f}")
    else:
        st.write('No data yet.  Press the "Run simulations" button above.')

if 'pc' in st.session_state:
    expand_sample = st.beta_expander("Expand to show the sample images and explanations", expanded=False)
    with expand_sample:
        make_sample()

explanation = st.beta_expander("Expand for more details about the process.", expanded=False)

with explanation:
    st.subheader("Explanations")
    st.markdown('''* All computations were made in Python.   You can download the source code from [Github](https://github.com/ChristopherDavisUCI/NFL2021-Simulation).
* You can duplicate and then edit this code in Deepnote.
* This website was made using [Streamlit](https://streamlit.io/).
* The plots were made using [Altair](https://altair-viz.github.io/). 
In the plots from your simulation (not the placeholder images), put your mouse over the bars to get more data.
* The thin black line represents the median win total for each team, so for example, if the line
passes through the color for 9 wins, that means that the simulation suggests that 9 is the most fair number for that team's over/under win total.
* Schedule data is taken from this excellent Kaggle dataset [NFL scores and betting data](https://www.kaggle.com/tobycrabtree/nfl-scores-and-betting-data).
* The default power ratings were computed based on the season-long lines and totals at [Superbook](https://co.superbook.com/sports) on July 16, 2021.
* You cannot edit both the combined power ratings and the separate power ratings.  If you switch from combined to separate or vice versa, it will delete any changes you made to the other.
* In the simulation, for each game, we compute the expected score for both teams using the power ratings.  (See the examples in the left-hand panel.) 
We then use a normal distribution with the expected score as the mean, and with 10 as the standard deviation.
We then round to the nearest integer, and replace any negative scores with zero.
* You can adjust the values of home field advantage and average team score at the bottom of the left-hand panel.
* No promises that my code is accurate.  (The hardest/most tedious part was implementing the tie-breaking procedures to determine playoff seedings. 
I believe all tie-breakers are incorporated except for the tie-breakers involving touchdowns.)
* Please report any bugs!
    ''')

follow = st.beta_expander("Possible follow-ups.", expanded=False)

with follow:
    st.subheader("Follow-ups with implementations in Deepnote")
    
    st.markdown('''* Given an over-under win total for a specific team, estimate what the fair odds should be.  (Warning.  If the over-under win total is 9, for example,
the fair odds for "over 9" does not correspond directly to the probability of the team winning 10 or more games, because pushes need to be treated differently from losses.)''')
    
    st.subheader("Follow-ups not yet implemented")

    st.markdown('''* Extend the simulations to include the playoffs.  Create charts showing which teams win the super bowl,
reach the super bowl, and reach the conference championship games most often.
* Our simulation does not take the order of games played into account.  Make a new version of the simulation which does.  For example, add some value to teams coming off a bye, or as another example,
let a team's power ranking evolve over the course of the season.
* I didn't think much about dealing with ties.  I wrote some ad hoc code that gets rid of most ties, 
with the home team slightly more likely to win in overtime than the road team.  (Without this ad hoc code, there were as many as ten ties per season.)  Come up with a more sophisticated solution.''')