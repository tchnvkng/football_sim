import time
from football_sim.football_sim import *

while True:
    calibrator = Calibrator('calibration.pkl', redo=False)
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
    _, _, fig = League.probability_grid()
    fig.savefig('prob_grid_end_of_season.png')
    t0 = pd.Timestamp('today')
    wd0 = t0.weekday()
    max_date = t0 + pd.Timedelta([3, 2, 1, 7, 6, 5, 4][wd0], 'd')
    League.process_simulation(max_date=max_date, verbose=False)
    fig.clear()
    _, _, fig = League.probability_grid()
    fig.savefig('prob_grid_{}.png'.format(str(max_date)[:10]))
    to_wait = 60
    while to_wait > 0:
        print('Waiting {:d} minutes.'.format(to_wait))
        time.sleep(60)
        to_wait -= 1
    print('Running again')
