from football_sim.all import Calibrator, Settings, Season
import os
import platform
import pandas as pd
import matplotlib.pyplot as plt
import time

if __name__ == '__main__':
    dates = []
    base_dir = os.path.dirname(os.path.realpath(__file__))
    if platform.node() == 'TokenBlack':
        # dates = dates + ['2019-09-30', '2019-10-30', '2019-11-30', '2019-12-30', '2020-01-30', '2020-02-28', '2020-03-30', '2020-04-30', '2020-05-30', '2020-06-30']
        # dates = dates + ['2020-07-01', '2020-07-08', '2020-07-09', '2020-07-10', '2020-07-11', '2020-07-12', '2020-07-13', '2020-07-14', '2020-07-15']
        output_dir = r'D:\output'
    else:
        output_dir = '/Users/manuel/Library/Mobile Documents/com~apple~CloudDocs/Documents/football_sim'
    print(output_dir)
    as_ofs = [pd.to_datetime(x) for x in dates]
    as_ofs.append(None)
    settings = Settings(os.path.join(base_dir, 'config.yaml'))
    calib = Calibrator(settings)
    year = 2020
    t = time.time()
    calib.download_all_data()
    print(time.time()-t,'download')

    t = time.time()
    calib.process_data()
    print(time.time() - t,'processed')
    t = time.time()

    # f = [f for f in calib.fixtures if f.league == 'BPL' and f.year == 2019 and 'Cry' in f.home_team.name and 'Manchester U' in f.away_team.name][0]
    # f.home_goals = 0
    # f.away_goals = 2
    # f.completed = True

    for as_of in as_ofs:
        print(as_of)
        for league in settings.domestic_leagues:
            t = time.time()
            calib.calibrate_teams(league, year, as_of=as_of)
            print(time.time() - t,league)
            season = Season(league, year, calib, use_home_advantage=False, base_folder=output_dir, as_of=as_of)
            print(league, 'home advantage', season.home_advantage)
            season.process_current_results()
            season.simulate_season(n_scenarios=100000)
            season.process_simulation()
            season.season_report(file_name='season_report')
            season.probability_grid(file_name='prob_grid')
            season.points_probability_grid(file_name='pnt_prob_grid')
            df = season.matches_remaining()
            for team_name in ['Manchester United','Liverpool','Chelsea','Manchester City','Manchester United Women','Salford City']:
                if team_name in season.teams:
                    print(league, team_name)
                    team = season.teams[team_name]
                    season.team_report(team, file_name=team_name.replace(' ', ''))
            plt.close(fig='all')

