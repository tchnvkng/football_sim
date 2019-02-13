import time
import os
import yaml
from football_sim.football_sim import *


if __name__ == "__main__":
    # base_dir = '/Volumes/iMacHD/OneDrive/football_sim'
    base_dir = './'
    league_info = yaml.load(open(os.path.join(base_dir, 'config.yaml'), 'r'))
    print(base_dir)
    calibrator = Calibrator('calibration.pkl', redo=False)
    calibrator.create_all_teams()
    calibrator.process_data()
    teams = calibrator.teams
    Teams = dict()
    League = dict()
    for name, info in league_info.items():
        print(name)
        if not os.path.isdir(os.path.join(base_dir, name)):
            os.mkdir(os.path.join(base_dir, name))
        lH, pH = calibrator.teams[name + 'Home'].means()
        lA, pA = calibrator.teams[name + 'Away'].means()
        home_advantage = np.array([lH - lA, pH / pA])
        Teams[name] = calibrator.get_teams_for_league(name)
        League[name] = Season(Teams[name], home_advantage=home_advantage,nr_cl=info['nr_cl'],nr_degr=info['nr_deg'])
        League[name].process_current_results(calibrator.get_current_results(name))
        lH, pH = calibrator.teams[name + 'Home'].means()
        lA, pA = calibrator.teams[name + 'Away'].means()
        home_advantage = np.array([lH - lA, pH / pA])
        print(home_advantage)
        League[name].simulate_season(n_scenarios=100000)
        League[name].season_report().to_html(os.path.join(base_dir, '{:s}/season_report.html'.format(name)))
        _, _, fig = League[name].probability_grid()
        fig.savefig(os.path.join(base_dir, '{:s}/prob_grid_end_of_season.png'.format(name)))
        fig.clear()
        t0 = pd.Timestamp('today')
        wd0 = t0.weekday()
        max_date = t0 + pd.Timedelta([3, 2, 1, 7, 6, 5, 4][wd0], 'd')
        League[name].process_simulation(max_date=max_date, verbose=False)
        _, _, fig = League[name].probability_grid()
        fig.savefig(os.path.join(base_dir, '{:s}/prob_grid_{}.png'.format(name, str(max_date)[:10])))
        fig.clear()
