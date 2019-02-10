import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from football_sim.football_sim import *
import os

calibrator = Calibrator('calibration.pkl', redo=True)
calibrator.create_all_teams()
calibrator.process_data()
teams = calibrator.teams

country = 'BPL'
lH, pH = calibrator.teams[country + 'Home'].means()
lA, pA = calibrator.teams[country + 'Away'].means()
home_advantage = np.array([lH - lA, pH / pA])
print(home_advantage)
Teams = calibrator.get_teams_for_league(country)
League = Season(Teams, home_advantage=home_advantage, nr_cl=4)
League.process_current_results(calibrator.get_current_results(country))
League.simulate_season(n_scenarios=100000)
League.season_report().to_html('bpl.html')



PremierLeague=League

ind = (PremierLeague.place_per_team[PremierLeague.team_id['Manchester United'],:]<=4)
PremierLeague.probability_grid(ind=ind)
imp = []
ref_team = 'Manchester United'
# ref_team='Liverpool'
# ref_team='Arsenal'
matches = []

for x in League.matches_to_sim:
    matches.append(x)
    pcl, _ = League.what_if(x, ref_team=ref_team, show_plot=False, place=4, or_better=True)
    # imp.append(pcl.max()/pcl.min())
    imp.append(pcl.std())
    # imp.append(1/pcl.min())
imp = np.array(imp)
matches = np.array(matches)
the_match = matches[imp.argmax()]
_, fig = League.what_if(the_match, ref_team=ref_team)
fig.set_size_inches(16, 9)
plt.tight_layout()
fig.savefig('mim.png')

fig = plt.figure(1)
fig.clear()
i_sort = (-imp).argsort()
matches = matches[i_sort]
imp = imp[i_sort]
xx = np.arange(20)
plt.barh(xx, imp[xx])
plt.yticks(xx, matches[xx], rotation=0);
plt.grid(True)
fig = plt.gcf()
fig.set_size_inches(16, 9)
plt.tight_layout()
plt.savefig('mims.png')
