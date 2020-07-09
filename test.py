from football_sim.all import Calibrator,Settings,Season
import os
import time

if __name__=='__main__':
    base_dir = './'

    settings = Settings(os.path.join(base_dir, 'config.yaml'))
    calib = Calibrator(settings)
    calib.download_all_data()
    calib.process_data()
    year = 2019
    league = 'BPL'
    calib.calibrate_teams(league, year)
    season = Season(league, year, calib, use_home_advantage=True)
    print('home advantage', season.home_advantage)

    season.process_current_results()
    season.simulate_season(n_scenarios=100000)
    season.process_simulation()
    season.season_report()


    my_team = calib.teams['BPL__BPLHome']
    my_team.plt()

    df = season.season_report()
    tras= 1