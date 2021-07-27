import streamlit as st
from numpy.random import default_rng
import pandas as pd
from sim_21season import simulate_reg_season, make_pr_custom
from make_standings import Standings
from make_charts import make_playoff_charts, make_win_charts, make_div_charts
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

max_reps = 2000

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
        st.write("These separate ratings are important for playoff tie-breakers, but for wins and losses, the overall power ratings are sufficient.")
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


st.markdown('''Based on your power ratings, we use Python to run many simulations of the 2021 NFL regular season, and then estimate answers to questions like:\n* How likely is Cleveland to get the no. 1 seed?  To win its division?
To make the playoffs?\n* How likely are the Steelers to win exactly 11 games?  To win 11 or more games?\n* How likely are the Patriots to finish 3rd in the AFC East?''')

st.write('''In each simulation, a random outcome is generated for all 272 regular season games.  The outcomes are based on the power ratings; you should customize these power ratings on the left.
(Warning.  Even if you set the absolute perfect power ratings, our simulation is too simple to provide true probabilities.  For example, 
our simulation does not account for the possibility of injuries during the season.)

If you're new to Python, I hope this app will make you want to try Python coding yourself.  All the tools I used are free, and most of them can be used online in Deepnote or Google Colab 
without installing anything on your computer.  Below I've posted some introductory YouTube playlists introducing the main tools I used (especially the Python library pandas), as well as links to Deepnote notebooks
where you can test out the tools yourself. 
[Sample Deepnote notebook](https://deepnote.com/project/NFL-2021-Simulation-XVJzHB7aTvGndVBV4CLYOA/%2FNFL2021-Simulation%2Fnfl_sim_byes.ipynb)

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

    #rank_dict = {div:{} for div in div_dict.keys()}
    rank_dict1 = {t:{} for t in teams}

    #for div in rank_dict.keys():
    #    for team_sort in permutations(div_dict[div]):
    #        rank_dict[div][team_sort] = 0

    for t in teams:
        for i in range(1,5):
            rank_dict1[t][i] = 0

    start = time.time()

    for i in range(reps):
        df = simulate_reg_season(pr)
        stand = Standings(df)

        p = stand.playoffs
        for conf in ["AFC","NFC"]:
            for j,t in enumerate(p[conf]):
                playoff_dict[conf][j+1][t] += 1
        for t in teams:
            team_outcome = stand.standings.loc[t]
            win_dict[t][team_outcome["Wins"]] += 1
            rank_dict1[t][team_outcome["Division_rank"]] += 1
        
        #for d in rank_dict.keys():
        #    rank_dict[d][tuple(stand.div_ranks[d])] += 1
        
        bar.progress((i+1)/reps)

    #for d in rank_dict.keys():
    #    rank_dict[d] = {i:j/reps for i,j in rank_dict[d].items()}

    #st.session_state["rd"] = rank_dict

    end = time.time()
    
    time_holder.write(f"{reps} simulations of the 2021 NFL regular season took {end - start:.1f} seconds.")


    playoff_charts = make_playoff_charts(playoff_dict)

    win_charts = make_win_charts(win_dict)

    div_charts = make_div_charts(rank_dict1)

    st.session_state['pc'] = playoff_charts
    st.session_state['wc'] = win_charts
    st.session_state['dc'] = div_charts


def make_ranking(df,col):
    return (-df[col]).rank()

def make_sample():
    st.header("Sample images:")
    if "pc" not in st.session_state:
        st.write('(To replace the sample images with real images, press the "Run simulations" button above.)')
    c_image, c_text = st.beta_columns(2)
    with c_image:
        
        st.image("images/pc_holder.png")
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
        st.image("images/wc_holder.png")
    with c_text:
        st.subheader("How to interpret the win total image.")
        st.markdown('''The AFC win total image shows the probability, according to our simulations, of different teams having a specific number of wins at the end of the 2021 regular season.
The thin black line represents the median win total for each team.\n\nFor example:
* The thin black vertical line passes through the median win total for each team.  The median win total for Kansas City is 12, the median win total for Cleveland is 10, while the median win total for Houston is 5.
* It looks like Baltimore has about a 48% chance of winning more than 10 games (the area to the left of the yellow bar), 
about a 32% chance of winning less than 10 games (the area to the right of the yellow bar), and about a 20% chance of winning exactly 10 games (the yellow bar).
* Pittsburgh has a 10.2% chance of winning exactly 11 games, and a 17.6% chance of winning at least 11 games.''')
    c_image, c_text = st.beta_columns(2)
    with c_image:
        st.image("images/dc_holder.png")
    with c_text:
        st.subheader("How to interpret the division ranks image.")
        st.markdown('''The division ranks image shows the probability of different teams finishing in specific ranks in their divisions.
\n\nFor example, according to 200 simulations:
* Buffalo has over a 60% chance of winning the AFC East, and the Jets have just under a 60% chance of finishing 4th in the AFC East.
* The Patriots have a 30.5% chance of finishing in third place in the AFC East.
\n\nThe teams are sorted in terms of how likely they are to win their division.  If all you care about is how likely is the team to win its division, then it's probably
more convenient to use the playoff seedings image.''')

if 'pc' in st.session_state:
    try:
        placeholder0.write(st.session_state['pc'])
        placeholder1.write(st.session_state['wc'])
    except:
        st.header("Simulation results")
        st.write(st.session_state['pc'])
        st.write(st.session_state['wc'])
else:
    make_sample()

df_rankings = pd.DataFrame({col:make_ranking(df_pr,col) for col in df_pr.columns})
    
radio_dict = {
    "YouTube": "YouTube playlists and Deepnote notebooks introducing the Python tools used.",
    "Rankings": "See the power rankings 1-32 based on the current values.",
    "Division": "Probabilities for different division ranks.",
    "Sample": "The sample images and explanations.",
    "Details": "More details about the process.",
    "Follow": "Possible follow-ups.",
}

info_choice = st.radio(
    'Options for more information',
    radio_dict.keys(),key="opt_radio",format_func=lambda k: radio_dict[k])

if info_choice == "Rankings":
    if pr_select == "Separate":
        st.write("(Click a column header to sort.)")
        st.dataframe(df_rankings.sort_values("Overall"),height=500)
    elif pr_select == "Combined":
        st.dataframe(df_rankings[["Overall"]].sort_values("Overall"),height=500)
elif info_choice == "Details":
    st.markdown('''* Warning! I'm not an expert on any of this material (not on Python, not on the NFL, not on random processes, not on simulating sports outcomes).
The point of this page is not to provide "true" probabilities.  My hope is that you will
think the process is interesting and will try to learn more about Python.  All of the resources I used are freely available.
* You can download the source code from [Github](https://github.com/ChristopherDavisUCI/NFL2021-Simulation).
* This website was made using [Streamlit](https://streamlit.io/).
* The plots were made using [Altair](https://altair-viz.github.io/). 
In the plots from your simulation (not the placeholder images), put your mouse over the bars to get more data.
* Schedule data was originally adapted from this excellent Kaggle dataset [NFL scores and betting data](https://www.kaggle.com/tobycrabtree/nfl-scores-and-betting-data).
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
elif info_choice == "Division":
    if "dc" in st.session_state:
        st.write(f"Based on {reps:.0f} simulations:")
        st.write(st.session_state["dc"])
    else:
        st.write('No data yet.  Press the "Run simulations" button above.')
elif info_choice == "YouTube":
    col_width = 10
    st.write("YouTube playlists introducing the main Python tools used.  For each playlist, there is an accompanying Deepnote notebook which can be edited.")
    col0,_,col1,_,col2 = st.beta_columns((col_width,1,col_width,1,col_width))
    with col0:
        st.subheader("52.4% and -110 odds")
        
    with col1:
        st.subheader("Calculating NFL power ratings from Spreads and Totals")
        
    with col2:
        st.subheader("Reading NFL point-spread data from html")
        
    col0,_,col1,_,col2 = st.beta_columns((col_width,1,col_width,1,col_width))
    with col0:
        st.write("Examples using the Python libraries NumPy, pandas, Altair, and scikit-learn")
    with col1:
        st.write("Using linear regression and point spreads/totals to compute offensive and defensive power ratings.")
    with col2:
        st.write("Using pandas and regular expressions to get data from html code.")

    col0,_,col1,_,col2 = st.beta_columns((col_width,1,col_width,1,col_width))
    with col0:
        st.markdown(''' <iframe width = 390 height = 220 src="https://www.youtube.com/embed/videoseries?list=PLHfGN68wSbbJ-z7mJ9F2OcbxjWp1A-efh" 
                title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; fullscreen;" allowfullscreen="allowfullscreen"></iframe>''',
                unsafe_allow_html=True)
    with col1:
        st.markdown('''<iframe width = 390 height = 220 src="https://www.youtube.com/embed/videoseries?list=PLHfGN68wSbbJHhPIdsRuAXEsNW4ACgtjL" 
        title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; fullscreen;" allowfullscreen="allowfullscreen"></iframe>''',
                unsafe_allow_html=True)
    with col2:
        st.markdown('''<iframe width = 390 height = 220 src="https://www.youtube.com/embed/videoseries?list=PLHfGN68wSbbIFxkUIdn57aLyfK40mOnBY" 
        title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; fullscreen;" allowfullscreen="allowfullscreen"></iframe>''',
                unsafe_allow_html=True)

    col0,_,col1,_,col2 = st.beta_columns((col_width,1,col_width,1,col_width))
    with col0:
        st.write("")
        st.markdown("Accompanying notebook in [Deepnote](https://deepnote.com/project/IntroNFL-5svnq68tSOGccRFoL48_jw/%2F524.ipynb) where you can try out the code yourself.")
    with col1:
        st.write("")
        st.markdown("Accompanying notebook in [Deepnote](https://deepnote.com/project/IntroNFL-5svnq68tSOGccRFoL48_jw/%2Fpower_ratings.ipynb) where you can try out the code yourself.")
    with col2:
        st.write("")
        st.markdown("Accompanying notebook in [Deepnote](https://deepnote.com/project/IntroNFL-5svnq68tSOGccRFoL48_jw/%2Freading_html.ipynb) where you can try out the code yourself.")
elif info_choice == "Sample":
    make_sample()
elif info_choice == "Follow":
    st.subheader("Follow-ups with implementations in Deepnote")
    
    st.markdown('''* Adapt the code so that teams coming off of a bye week have slightly boosted power ratings.
[Sample solution in Deepnote](https://deepnote.com/project/NFL-2021-Simulation-XVJzHB7aTvGndVBV4CLYOA/%2FNFL2021-Simulation%2Fnfl_sim_byes.ipynb)''')
    
    st.subheader("Follow-ups not yet implemented")

    st.markdown('''* Given an over-under win total for a specific team, estimate what the fair odds should be.  (Warning.  If the over-under win total is 9, for example,
the fair odds for "over 9" does not correspond directly to the probability of the team winning 10 or more games, because pushes need to be treated differently from losses.)
* Extend the simulations to include the playoffs.  Create charts showing which teams win the super bowl,
reach the super bowl, and reach the conference championship games most often.
* One weakness of our simulation is that a team has the same power rating throughout the entire season.  In fact, the order in which games are played has no impact on our simulation.
Adapt the code so that the power ratings evolve over the course of the season.
* I didn't think much about dealing with ties.  I wrote some ad hoc code that gets rid of most ties, 
with the home team slightly more likely to win in overtime than the road team.  (Without this ad hoc code, there were as many as ten ties per season.)  Come up with a more sophisticated solution.''')

