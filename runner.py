from football_sim.all import Calibrator, Settings, Season
import os
import platform

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.realpath(__file__))
    if platform.node() == 'Speedy':
        output_dir = r'D:\output'
    else:
        output_dir = '/Users/manuel/Library/Mobile Documents/com~apple~CloudDocs/Documents/football_sim'

    settings = Settings(os.path.join(base_dir, 'config.yaml'))
    calib = Calibrator(settings)
    year = 2019
    calib.download_all_data()
    calib.process_data()

    for league in settings.domestic_leagues:
        calib.calibrate_teams(league, year)
        season = Season(league, year, calib, use_home_advantage=False, base_folder=output_dir)
        print(league, 'home advantage', season.home_advantage)
        season.process_current_results()
        season.simulate_season(n_scenarios=100000)
        season.process_simulation()
        season.season_report(file_name='season_report.html', add_date_to_file_name=True)
        season.probability_grid(file_name='prob_grid.png', add_date_to_file_name=True)
        season.points_probability_grid(file_name='pnt_prob_grid.png', add_date_to_file_name=True)
        df = season.matches_remaining()
        for team_name in ['Manchester United', 'Chelsea', 'Leicester City', 'Juventus', 'Real Madrid', 'Barcelona', 'Lazio','Atalanta','Leeds United']:
            if team_name in season.teams:
                print(league,team_name)
                team = season.teams[team_name]
                season.team_report(team, file_name=team_name.replace(' ', '') + '.png', add_date_to_file_name=True)
