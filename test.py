from football_sim.all import Calibrator,Settings,Season
import os
import time

if __name__=='__main__':
    base_dir = './'

    settings = Settings(os.path.join(base_dir, 'config.yaml'))
    calib = Calibrator(settings)
    calib.download_all_data()
    calib.process_data()
    season = 2019
    league = 'BPL'
    calib.calibrate_teams(league, season)
    season = Season(league, season, calib, use_home_advantage=True)
    print('home advantage', season.home_advantage)

    season.process_current_results()
    season.simulate_season(n_scenarios=100000)
    season.process_simulation()
    season.season_report()

    df = season.season_report()
    tras= 1