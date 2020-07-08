from football_sim.football_sim import *
# from football_sim.footystats import get_data_and_merge
# if __name__ == "__main__":
print('Start ', pd.Timestamp('today'))
# base_dir = '/Users/manuel/Downloads/OneDrive-2019-12-30'
# get_data_and_merge()
base_dir = '/Users/manuel/Library/Mobile Documents/com~apple~CloudDocs/Documents/football_sim'

settings = Settings(os.path.join(base_dir, 'config.yaml'))



calibrator = Calibrator('2019-2020.pkl', settings,2019,redo=True)
calibrator.create_all_teams()
calibrator.process_data(verbose=True)


t0 = pd.Timestamp('today')
for name, info in settings.league_info.items():
    print(name)
    #try:
    if not os.path.isdir(os.path.join(base_dir, name)):
        os.mkdir(os.path.join(base_dir, name))
    league = Season(name, calibrator,use_home_advantage=False)
    print('home advantage',league.home_advantage)
    current_results = calibrator.get_current_results(name)
    league.process_current_results(current_results)
    league.simulate_season(n_scenarios=100000)



    max_date_indices = league.remaining_date_indices[1:]
    for mdi in max_date_indices:
        league.process_simulation(max_date_index=mdi, verbose=False)
        _, _, fig = league.probability_grid(title='{:s}|{:0.0f}'.format(t0.strftime('%Y-%m-%d'),mdi))
        fig.savefig(os.path.join(base_dir, '{:s}/{:s}_prob_grid_{:0.0f}.png'.format(name, t0.strftime('%Y-%m-%d'), mdi)))
        fig.clear()
    league.process_simulation(verbose=False)
    _, _, fig = league.probability_grid(title='{:s}|EOS'.format(t0.strftime('%Y-%m-%d')))
    fig.savefig(os.path.join(base_dir, '{:s}/{:s}_prob_grid_EOS.png'.format(name, t0.strftime('%Y-%m-%d'))))
    fig.clear()
    sr_fn = '{:s}/{:s}_season_report.html'.format(name, t0.strftime('%Y-%m-%d'))
    sr_fn = os.path.join(base_dir, sr_fn)
    print(sr_fn)
    league.season_report().to_html(sr_fn)
    sr_fn = '{:s}/{:s}_season_report.xlsx'.format(name, t0.strftime('%Y-%m-%d'))
    sr_fn = os.path.join(base_dir, sr_fn)
    league.season_report().to_excel(sr_fn)

    sr_fn = '{:s}/{:s}_matches.html'.format(name, t0.strftime('%Y-%m-%d'))
    sr_fn = os.path.join(base_dir, sr_fn)
    league.matches_remaining().to_html(sr_fn)
    _, _, fig = league.points_probability_grid()
    fig.savefig(os.path.join(base_dir, '{:s}/{:s}_points_prob_grid.png'.format(name, t0.strftime('%Y-%m-%d'))))
    fig.clear()




    #except Exception as err:
    #    print(err)
print('End ', pd.Timestamp('today'))
