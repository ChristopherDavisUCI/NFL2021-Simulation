import pandas as pd
import altair as alt

df = pd.read_csv("schedules/schedule21.csv")
div_series = pd.read_csv("data/divisions.csv",squeeze=True,index_col=0)
teams = sorted(list(set(df.team_home)))
conf_teams = {}
for conf in ["AFC","NFC"]:
    conf_teams[conf] = [t for t in teams if div_series[t][:3]==conf]

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

def make_playoff_charts(total_dict):

    reps = sum(total_dict["AFC"][1].values())

    odds_dict = {"Proportion":"Odds", "Make_playoffs": "Odds_Make_playoffs", "Equal_better":"Odds_Equal_better"}

    # Trying to match the Tableau10 colors in the order I want.
    #color_dict = dict(zip(["purple","yellow","green","teal","red","orange","blue"],["#b07aa1","#EDC948","#59A14F","#76B7B2","#E15759","#F28E2B","#4E79A7"]))
    color_dict = {"blue": "#5778a4", "orange": "#e49444", "red": "#d1615d", "teal": "#85b6b2", "green": "#6a9f58", "yellow": "#e7ca60", "purple": "#a87c9f",}

    chart_dict = {}

    for conf in ["AFC","NFC"]:

        playoff_dicts = total_dict[conf]
        
        source = pd.DataFrame([(k,t,playoff_dicts[k][t]/reps) for k in playoff_dicts.keys() for t in conf_teams[conf]],
            columns = ["Seed","Team","Proportion"])
        
        for a,b in source.groupby("Team"):
            source.loc[source["Team"] == a, "Make_playoffs"] = b.Proportion.sum()
            source.loc[source["Team"] == a,"Equal_better"] = b.Proportion.cumsum()

        #source["Equal_better"] += source["Proportion"]

        for c in odds_dict.keys():
            source[odds_dict[c]] = source[c].map(prob_to_odds)

        ordering = sorted(conf_teams[conf],key=lambda t: sum([playoff_dicts[i][t] for i in playoff_dicts.keys()]),reverse=True)
        ordering_seeds = list(range(7,0,-1))


        c = alt.Chart(source).mark_bar().encode(
            x=alt.X('Team',sort = ordering),
            y=alt.Y('Proportion',scale=alt.Scale(domain=[0,1]),sort=ordering_seeds),
            tooltip = [alt.Tooltip('Team'),alt.Tooltip('Seed', format=".0f"), 
                        #alt.Tooltip('Proportion', format=".3f"),
                        alt.Tooltip("prob_odds:N",title="Proportion"),
                        alt.Tooltip("equal_odds:N",title="Equal or better"),
                        alt.Tooltip("playoffs_odds:N",title="Make playoffs")],
            color=alt.Color('Seed:N', scale=alt.Scale(domain=[7,6,5,4,3,2,1],range=[color_dict[color] for color in ["purple","yellow","green","teal","red","orange","blue"]])),
            order=alt.Order(
                    'Seed:N',
                    sort='ascending'
            )).transform_calculate(
                prob_odds="format(datum.Proportion, ',.3f')+' (' +datum.Odds+')'",
                equal_odds="format(datum.Equal_better, ',.3f')+' (' +datum.Odds_Equal_better+')'",
                playoffs_odds="format(datum.Make_playoffs, ',.3f')+' (' +datum.Odds_Make_playoffs+')'"
            ).properties(
                title=f"{conf} playoff seedings",
                width=500,
                height=500,
            )
        
        chart_dict[conf] = c
        
    playoff_charts = alt.hconcat(*chart_dict.values()).resolve_scale(
        color='independent'
    ).properties(
        title=f"Based on {reps} simulations:"
    )

    return playoff_charts

def make_win_charts(win_dict):
    odds_dict2 = {"Proportion":"Odds", "Equal_higher":"Odds_Equal_higher"}

    reps = sum(win_dict["ARI"].values())

    win_charts = {}
    for conf in ["AFC","NFC"]:
        source = pd.DataFrame([(w,t,win_dict[t][w]/reps) for t in conf_teams[conf] for w in range(18)],
                    columns = ["Wins","Team","Proportion"])
        
        for a,b in source.groupby("Team"):
            source.loc[source["Team"] == a,"Equal_higher"] = 1 - b.Proportion.cumsum()
        
        source["Equal_higher"] += source["Proportion"]

        for c in odds_dict2.keys():
            source[odds_dict2[c]] = source[c].map(prob_to_odds)
        
        ordering = sorted(conf_teams[conf],
            key = lambda t: sum([a*b for a,b in win_dict[t].items()])/reps, reverse = True)

        ordering_wins = list(range(18,-1,-1))

        c = alt.Chart(source).mark_bar().encode(
                y=alt.Y('Team',sort = ordering),
                x=alt.X('Proportion',scale=alt.Scale(domain=[0,1])),
                tooltip = [alt.Tooltip("Team"),
                    alt.Tooltip('Wins', format=".0f"),
                    alt.Tooltip('prob_odds:N',title="Proportion"),
                    alt.Tooltip('equal_odds:N',title="Equal or higher")],
                color=alt.Color('Wins:N', scale=alt.Scale(scheme='tableau20'),
                            sort = ordering_wins),
                order=alt.Order(
                    'Wins:N',
                    sort='descending'
                )
            ).transform_calculate(
                prob_odds="format(datum.Proportion, ',.3f')+' (' +datum.Odds+')'",
                equal_odds="format(datum.Equal_higher, ',.3f')+' (' +datum.Odds_Equal_higher+')'"
            ).properties(
                title=f"{conf} win totals",
                width=500,
                height=500,
            )

        overlay = pd.DataFrame({'Proportion': [0.5]})
        vline = alt.Chart(overlay).mark_rule(color='black', strokeWidth=.6).encode(x='Proportion:Q')

        win_charts[conf] = c + vline


    win_totals = alt.hconcat(*win_charts.values()).resolve_scale(
        color='independent'
    ).properties(
        title=f"Based on {reps} simulations:"
    )

    return win_totals

def custom_sort(rd,div):
    div_teams = list(div_series[div_series==div].index)
    return sorted(div_teams,key = lambda t: rd[t][1], reverse=True)

def make_div_charts(rd):

    reps = sum(rd["ARI"].values())

    source = pd.DataFrame([(t,j,rd[t][j]/reps,div_series[t],0) for t in teams for j in range(1,5)],
            columns = ["Team","Rank","Proportion","Division","Odds"])

    source["Odds"] = source["Proportion"].map(prob_to_odds)

    output_dict = {}

    for div in sorted(list(set(div_series))):

        output_dict[div] = alt.Chart(source.query("Division==@div")).mark_bar().encode(
            x = alt.X("Rank:O",title="Place"),
            y = alt.Y("Proportion", scale=alt.Scale(domain=(0,1))),
            color = alt.Color("Proportion",scale=alt.Scale(scheme="lighttealblue", domain=(0,1))),
            column = alt.Column("Team:N", sort=custom_sort(rd,div)),
            tooltip = ["Team","Division","Rank",alt.Tooltip('Proportion', format=".3f"),alt.Tooltip('Odds:N')]
        ).properties(
            title = div,
            width = 100,
            height = 250
        )
    return alt.vconcat(*output_dict.values()).resolve_scale(
            color='independent'
        ).configure_concat(
            spacing=50
        )