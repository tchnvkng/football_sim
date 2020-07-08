from football_sim.all import Calibrator,Settings,Season
import os
import time

if __name__=='__main__':
    base_dir = '/Users/manuel/Desktop/football_sim'

    settings = Settings(os.path.join(base_dir, 'config.yaml'))

    t0 = time.time()
    calib = Calibrator(settings)
    print('create',time.time()-t0)
    t0 = time.time()
    calib.download_all_data()
    print('download',time.time() - t0)
    t0 = time.time()
    calib.process_data()
    print('process',time.time() - t0)
    t0 = time.time()
    season = 2019
    league = 'BPL'
    [f.update_teams(update_forecast=True) for f in calib.fixtures.values() if f.season >= season-1 and f.league == league]
    season = Season(league, season,calib, use_home_advantage=False)
    print('home advantage', season.home_advantage)

    season.process_current_results()
    season.simulate_season(n_scenarios=100000)
    season.process_simulation()
    df = season.season_report()
    tras= 1