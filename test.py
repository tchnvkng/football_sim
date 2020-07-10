from football_sim.all import Calibrator, Settings, Season
import os
import time


if __name__ == '__main__':
    base_dir = './'

    settings = Settings(os.path.join(base_dir, 'config.yaml'))
    calib = Calibrator(settings)
    year = 2019
    calib.download_all_data()
    calib.process_data()

    for league in settings.domestic_leagues:
        calib.calibrate_teams(league, year)
        season = Season(league, year, calib, use_home_advantage=False, base_folder=base_dir)
        print(league, 'home advantage', season.home_advantage)
        season.process_current_results()
        season.simulate_season(n_scenarios=100000)
        season.process_simulation()
        season.season_report(file_name='season_report.html', add_date_to_file_name=True)
        season.probability_grid(file_name='prob_grid.png', add_date_to_file_name=True)
        season.points_probability_grid(file_name='pnt_prob_grid.png', add_date_to_file_name=True)
        df = season.matches_remaining()
        if league == 'BPL':
            bpl = season
            my_team = season.teams['Manchester United']
            bpl.team_report(my_team,file_name='mufc.png',add_date_to_file_name=True)


    df = season.season_report()
    tras = 1
