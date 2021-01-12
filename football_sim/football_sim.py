import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pickle
import os
import copy
import http.client
import yaml
import time

plt.rcParams['figure.figsize'] = [16, 9]


class Settings:
    def __init__(self,settings_file):
        self.league_info = yaml.safe_load(open(settings_file, 'r'))
        self.domestic_leagues = list(self.league_info.keys())
        self.eu_leagues = ['UCL', 'UEL']
    def get_info(self,name):
        if name in self.league_info:
            return self.league_info[name]




def add_match(data, home, home_goals, away, away_goals, the_date=pd.to_datetime('today')):
    if data.index.shape[0] > 0:
        max_ind = data.index.max()
    else:
        max_ind = 0
    a = pd.DataFrame({'Date': the_date, 'HomeTeam': home, 'AwayTeam': away, 'FTHG': home_goals, 'FTAG': away_goals}, index=[max_ind + 1])
    a = a[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']]
    return data.append(a)


class Fixture:
    def __init__(self, row,settings):
        self.domestic_leagues = settings.domestic_leagues
        self.eu_leagues = settings.eu_leagues
        self.league = row['League']
        self.home_team_name = row['HomeTeam']
        self.away_team_name = row['AwayTeam']
        self.league_team_of_home_team = None
        self.league_team_of_away_team = None
        self.league_home_team = None
        self.league_away_team = None
        self.home_team = None
        self.away_team = None
        self.date = row['Date']
        # a = 1/10
        # f = lambda x: (1- np.exp(-a*x))/a
        f = lambda x: x
        self.home_goals = f(row['FTHG'])
        self.away_goals = f(row['FTAG'])
        self.home_metrics = [row['xg1'],row['nsxg1'],row['adj_score1']]
        self.away_metrics = [row['xg2'], row['nsxg2'], row['adj_score2']]
        if np.isnan(self.home_metrics).any() or np.isnan(self.away_metrics).any():
            self.home_ag = self.home_goals
            self.away_ag = self.away_goals
        else:
            # self.home_ag = self.home_goals
            # self.away_ag = self.away_goals
            self.home_ag = np.mean(self.home_metrics)
            self.away_ag = np.mean(self.away_metrics)


        self.id = ('_'.join([self.league, self.home_team_name, self.away_team_name, self.date.strftime('%Y-%m-%d')])).replace(' ', '').lower()

    def set_teams(self, team_dict):
        self.home_team = team_dict[self.home_team_name]
        self.away_team = team_dict[self.away_team_name]
        self.league_team_of_home_team = team_dict[self.home_team.country]
        self.league_team_of_away_team = team_dict[self.away_team.country]
        self.league_home_team = team_dict[self.home_team.country + 'Home']
        self.league_away_team = team_dict[self.away_team.country + 'Away']
    def __repr__(self):
        h = [self.home_goals] +  self.home_metrics
        a = [self.away_goals] + self.away_metrics
        i = ['goals','xg','nsxg','ag']
        df = pd.DataFrame({'metric':i,self.home_team_name:h,self.away_team_name:a})
        return df.__repr__()


class Calibrator:
    def __init__(self, file_name, settings,season , old_teams=dict(), redo=False, league_start=None, calibration_start=None):
        self.settings = settings
        self.season = season
        self.file_name = file_name
        self.teams = dict()
        self.old_teams = old_teams
        self.raw_data = None
        t0 = pd.Timestamp('today')

        if t0.month >= 7:
            t_default_start = pd.Timestamp(t0.year, 7, 1)
        else:
            t_default_start = pd.Timestamp(t0.year - 1, 7, 1)
        if league_start is None:
            self.league_start = t_default_start
        else:
            self.league_start = pd.Timestamp(league_start)
        if calibration_start is None:
            self.calibration_start = t_default_start
        else:
            self.calibration_start = pd.Timestamp(calibration_start)
        self._teams_created = False
        self.domestic_leagues = settings.domestic_leagues
        self.eu_leagues = settings.eu_leagues
        self.processed_matches = dict()
        if redo:
            print('Force recalibrate')
        if os.path.isfile(file_name):
            print(file_name, ' exists.')
        else:
            print(file_name, ' does not exists')

        if os.path.isfile(file_name) and not redo:
            print('file exists, loading')
            self.load()
            print(len(self.processed_matches))
        else:
            self.save()

    def load(self):
        with open(self.file_name, 'rb') as input:
            self.teams = pickle.load(input)
            self.processed_matches = pickle.load(input)
            self.raw_data = pickle.load(input)

    def save(self):
        if os.path.isfile(self.file_name):
            os.remove(self.file_name)

        with open(self.file_name, 'wb') as output:
            pickle.dump(self.teams, output, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.processed_matches, output, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.raw_data, output, pickle.HIGHEST_PROTOCOL)

    def create_team(self, team_name, country):
        if team_name not in self.teams:
            if team_name in self.old_teams:
                self.teams[team_name] = copy.deepcopy(self.old_teams[team_name])
                self.teams[team_name].forget()
            else:
                self.teams[team_name] = Team(name=team_name, country=country)

    def download_all_data(self):
        f = lambda x: "".join([y[0] for y in x.upper().split()])
        df = pd.read_csv('https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv')
        # df = pd.read_csv('spi_matches_plus.csv')
        df = df.rename(columns={'team1': 'HomeTeam', 'team2': 'AwayTeam', 'score1': 'FTHG', 'score2': 'FTAG',
                                'league': 'League','season':'Season'})
        ind = (df['FTAG'].isnull() | df['FTHG'].isnull())
        df.loc[ind, 'FTAG']=-100
        df.loc[ind, 'FTHG'] = -100
        ind = df['Season'].isnull()
        df.loc[ind, 'Season'] = -100
        df['Season'] = df['Season'].astype(int)
        df['FTAG'] = df['FTAG'].astype(int)
        df['FTHG'] = df['FTHG'].astype(int)
        df['Date'] = pd.to_datetime(df['date'])
        df.to_csv('spi_matches.csv', index=False)
        df = df[['Date','Season','League', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'xg1', 'xg2', 'nsxg1',
                 'nsxg2','adj_score1','adj_score2']]
        # ind = (df['Date'] > self.calibration_start)
        ind = df['Season'] == self.season
        df = df.loc[ind]
        df['League'] = df['League'].apply(f)
        # replace_dict = {'\xfc': 'ue', '\xe9': 'e'}
        #for c1, c2 in replace_dict.items():
        #    pass
        #    #df['HomeTeam'] = df['HomeTeam'].apply(lambda x: x.replace(c1, c2))
        #    #df['AwayTeam'] = df['AwayTeam'].apply(lambda x: x.replace(c1, c2))
        self.raw_data = df

    def get_current_results(self, league, only_done=False):
        if self.raw_data is None:
            self.download_all_data()
        ind = self.raw_data['League'] == league
        # ind = ind & (self.raw_data['Date'] > self.league_start)
        ind = ind & (self.raw_data['Season'] == self.season)
        if only_done:
            ind = ind & (self.raw_data['FTAG'] >= 0)
            ind = ind & (self.raw_data['FTHG'] >= 0)

        return self.raw_data.loc[ind]

    def get_teams_for_league(self, _league):
        ind = self.raw_data['League'] == _league
        ind = ind & (self.raw_data['Season'] == self.season)
        # ind = ind & (self.raw_data['Date'] >= self.league_start)
        _team_names = np.unique(self.raw_data.loc[ind, ['HomeTeam', 'AwayTeam']].values)
        return {x: self.teams[x] for x in _team_names}

    def create_all_teams(self):
        if self.raw_data is None:
            self.download_all_data()
        for _league in self.domestic_leagues:
            self.create_team(_league, 'UEFA')
            self.create_team(_league + 'Home', _league + '0')
            self.create_team(_league + 'Away', _league + '0')
        ind = self.raw_data['League'].apply(lambda x: x in self.domestic_leagues)
        # ind = ind & (self.raw_data['Date'] > self.calibration_start)
        for _, row in self.raw_data.loc[ind, ['League', 'HomeTeam', 'AwayTeam']].iterrows():
            home_team_name = row['HomeTeam']
            away_team_name = row['AwayTeam']
            league = row['League']
            self.create_team(home_team_name, league)
            self.create_team(away_team_name, league)
        self._teams_created = True

    def process_data(self, update_params=True, verbose=False):
        self.download_all_data()
        if not self._teams_created:
            self.create_all_teams()
        ind = self.raw_data['League'].apply(lambda x: x in self.domestic_leagues + self.eu_leagues)
        # ind = ind & (self.raw_data['Date'] > self.calibration_start)
        ind = ind & (self.raw_data['Season'] == self.season)
        ind = ind & (self.raw_data['FTAG'] >= 0)
        ind = ind & (self.raw_data['FTHG'] >= 0)
        t0 = pd.Timestamp('today')
        t1 = t0 - pd.Timedelta('270 days')
        print(t0,t1)
        ind = ind & (self.raw_data['Date'] > t1)
        for index, row in self.raw_data.loc[ind].iterrows():
            fixture = Fixture(row,self.settings)
            try:
                if fixture.id not in self.processed_matches and update_params:
                    if not (np.isnan(fixture.home_goals) or np.isnan(fixture.away_goals)):
                        if verbose:
                            print((fixture.date, fixture.home_team_name, fixture.away_team_name, fixture.home_goals, fixture.away_goals))
                        if fixture.league in self.domestic_leagues:
                            fixture.set_teams(self.teams)
                            fixture.home_team.scored_against(fixture.away_team, fixture.home_ag)
                            fixture.away_team.scored_against(fixture.home_team, fixture.away_ag)
                            fixture.league_home_team.scored_against(fixture.league_away_team, fixture.home_ag)
                            fixture.league_away_team.scored_against(fixture.league_home_team, fixture.away_ag)
                        elif fixture.league in self.eu_leagues and fixture.home_team_name in self.teams and fixture.away_team_name in self.teams:
                            fixture.set_teams(self.teams)
                            fixture.league_team_of_home_team.scored_against(fixture.league_team_of_away_team, fixture.home_ag)
                            fixture.league_team_of_away_team.scored_against(fixture.league_team_of_home_team, fixture.away_ag)
                    self.processed_matches[fixture.id] = fixture
            except Exception as err:
                print(err)

        self.save()


class Team(object):
    def __init__(self, name='team name', country='SH'):
        self.name = name
        self.country = country
        self.lmbd_set = np.linspace(0, 10, 101)
        self.p = self.lmbd_set * 0 + 1
        self.p = self.p / self.p.sum()
        self.tau_set = np.linspace(0, 1, 101)
        self.q = self.tau_set * 0 + 1
        self.q = self.q / self.q.sum()

    def simplify(self, threshold=1e-10):
        ind = self.p > threshold
        self.lmbd_set = self.lmbd_set[ind]
        self.p = self.p[ind]
        ind = self.q > threshold
        self.tau_set = self.tau_set[ind]
        self.q = self.q[ind]
        self.normalize()

    def normalize(self):
        self.p = self.p / self.p.sum()
        self.q = self.q / self.q.sum()

    def forget(self, p_mix=0.5):
        self.p = (1 - p_mix) * self.p + p_mix / self.p.shape[0]
        self.q = (1 - p_mix) * self.q + p_mix / self.q.shape[0]
        self.normalize()

    def outcomes_vs(self, other_team, n_scenarios=int(1e5), home_advantage=1):
        g = np.zeros([n_scenarios, 2])
        g[:, 0], g[:, 1], _ = self.vs(other_team, n=n_scenarios, home_advantage=home_advantage)
        u, c = np.unique(g, axis=0, return_counts=True)
        loc = (-c).argsort()
        u = u[loc, :]
        c = c[loc]
        x = np.arange(u.shape[0])

        ind = [u[:, 0] > u[:, 1], u[:, 0] == u[:, 1], u[:, 0] < u[:, 1]]

        p = 100 * c / n_scenarios
        lab = [self.name + ' win', 'draw', other_team.name + ' win']
        col = ['green', 'yellow', 'red']
        for _ind, _l, _c in zip(ind, lab, col):
            y = p[_ind]
            plt.bar(x[_ind], y, label='{:s}: {:0.1f}%'.format(_l, y.sum()))
        plt.xticks(x, u, rotation='vertical');
        plt.legend()
        plt.xlim(-0.5, x[p > 0.5].max() + 0.5)
        plt.grid()
        gmean = g.mean(axis=0)
        plt.title('{} ({:.2f}) vs {} ({:.2f})'.format(self.name, gmean[0], other_team.name, gmean[1]))

    def vs(self, other_team, n=int(1e4), home_advantage=1):
        lH = np.random.choice(self.lmbd_set, size=n, p=self.p)  * np.random.choice(other_team.tau_set, size=n, p=other_team.q)*home_advantage
        gH = np.random.poisson(lH)
        lA = np.random.choice(other_team.lmbd_set, size=n, p=other_team.p) * np.random.choice(self.tau_set, size=n, p=self.q)
        gA = np.random.poisson(lA)
        match_des = self.name + ' vs ' + other_team.name
        return gH, gA, match_des

    def plt(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 2)
        l, t = self.means()
        p1 = ax[0].plot(self.lmbd_set, self.p, label=self.name + ' off: {:0.2f}'.format(l))
        ax[1].plot(self.tau_set, self.q, c=p1[0].get_color(), label=self.name + ' def: {:0.2f}'.format(t))
        ax[0].legend()
        ax[0].grid(True)
        ax[1].legend()
        ax[1].grid(True)
        # l, t = self.means()
        # ax[0].set_title('lambda: {:0.2f}'.format(l))
        # ax[1].set_title('tau: {:0.2f}'.format(t))
        return ax

    def means(self):
        return self.p.dot(self.lmbd_set), self.q.dot(self.tau_set)

    def scored_against(self, other, k):
        x0 = 3
        x1 = 10
        y1 = 7
        b = 0.5 * (y1 - x1) / (x1 - x0)
        c = 1 + b
        a = -b * x0
        f_abs = lambda x: np.sqrt(x ** 2 + 1)
        # k = a + b * f_abs(k - x0) + c * k
        lmb_times_tau = self.lmbd_set * other.tau_set[:, np.newaxis]
        new_p = ((np.exp(-lmb_times_tau) * (lmb_times_tau ** k)).T * other.q).sum(axis=1) * self.p
        self.p = new_p / new_p.sum()
        new_q = ((np.exp(-lmb_times_tau) * (lmb_times_tau ** k)) * self.p).sum(axis=1) * other.q
        other.q = new_q / new_q.sum()



def p_plot(x):
    a = x.min()
    b = x.max()
    xx = np.arange(a, b + 1)
    yy = xx * 0
    for _i in range(xx.shape[0]):
        yy[_i] = (x == xx[_i]).sum()
    yy = 100 * yy / yy.sum()
    return xx, yy


class Season:
    def __init__(self, name, calibrator, nr_cl=4, nr_degr=3, use_home_advantage = True):
        self.name = name
        self.teams = calibrator.get_teams_for_league(name)
        if use_home_advantage:
            lH, pH = calibrator.teams[self.name + 'Home'].means()
            lA, pA = calibrator.teams[self.name + 'Away'].means()
            self.home_advantage = lH * pA / (lA * pH)
        else:
            self.home_advantage = 1
        info = calibrator.settings.get_info(self.name)
        self.nr_cl = info['nr_cl']
        self.nr_degr = info['nr_deg']
        self.nr_teams = len(self.teams)
        self.all_matches = {home + ' v ' + away: {'Done': False, 'Home': home, 'Away': away,'Date': pd.Timestamp('today'),'eff_date_index':0,'eff_date': pd.Timestamp('today')} for home in self.teams for away
                            in self.teams if home != away}
        self.matches_to_sim = self.all_matches
        self.current_goals = dict()
        self.current_goals_against = dict()
        self.current_points = dict()
        self.played=dict()
        self.simulated_home_goals = None
        self.simulated_away_goals = None
        self.simulated_home_points = None
        self.simulated_away_points = None
        self.team_id = dict()
        self.inv_team_id = dict()
        i = 0
        for _team in self.teams:
            self.team_id[_team] = i
            self.inv_team_id[i] = _team
            self.current_goals[_team] = 0
            self.current_goals_against[_team] = 0
            self.current_points[_team] = 0
            self.played[_team] = 0
            i += 1
        self.simulation_done = False
        self.simulation_processed = False

    def process_current_results(self, data):
        data = data.sort_values(by='Date')

        #n_played = np.zeros(self.nr_teams)
        all_teams = list(self.teams.keys())
        cum_played = {x: 0 for x in all_teams}
        eff_date_index = 1
        eff_date = data['Date'].min()
        match_cache = []
        for index, row in data.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            match = home_team + ' v ' + away_team
            match_cache.append(match)
            if home_team in cum_played:
                cum_played[home_team] += 1
            else:
                cum_played[home_team] = 0
            if away_team in cum_played:
                cum_played[away_team] += 1
            else:
                cum_played[away_team] = 0
            #for i, x in enumerate(all_teams):
            #    n_played[i] = cum_played[x]
            n_played = list(cum_played.values())
            if np.std(n_played) == 0:
                eff_date_index = int(n_played[0])
                eff_date = row['Date']
                for _match in match_cache:
                    self.all_matches[_match]['eff_date_index'] = eff_date_index
                    self.all_matches[_match]['eff_date'] = eff_date
                match_cache = []




            home_goals = row['FTHG']
            away_goals = row['FTAG']
            self.all_matches[match]['Date'] = row['Date']

            if (home_goals >= 0) & (away_goals >= 0):
                if not (np.isnan(home_goals) or np.isnan(
                        away_goals)) and home_team in self.teams and away_team in self.teams:
                    self.current_goals[home_team] += home_goals
                    self.current_goals[away_team] += away_goals
                    self.current_goals_against[home_team] += away_goals
                    self.current_goals_against[away_team] += home_goals
                    self.played[home_team] += 1
                    self.played[away_team] += 1

                    if home_goals > away_goals:
                        self.current_points[home_team] += 3
                    elif home_goals < away_goals:
                        self.current_points[away_team] += 3
                    else:
                        self.current_points[home_team] += 1
                        self.current_points[away_team] += 1
                    self.all_matches[match]['Done'] = True

        self.matches_to_sim = {x: self.all_matches[x] for x in self.all_matches if not self.all_matches[x]['Done']}
        self.remaining_date_indices = np.sort(np.unique([x['eff_date_index'] for x in self.matches_to_sim.values()]))
        self.remaining_eff_date = np.sort(np.unique([x['eff_date'] for x in self.matches_to_sim.values()]))

    def matches_remaining(self):
        home = []
        away = []
        date = []
        home_win = []
        draw = []
        away_win = []
        average_home = []
        average_away = []
        n_sim = self.simulated_away_goals.shape[1]
        for match_name, match_info in self.matches_to_sim.items():
            i = match_info['id']
            home_ = match_info['Home']
            home.append(home_)
            away_ = match_info['Away']
            away.append(away_)
            date_ = match_info['Date']
            date.append(date_)
            hg = self.simulated_home_goals[i, :]
            ag = self.simulated_away_goals[i, :]
            # home_win.append('{:0.0f}%'.format(100*(hg>ag).sum()/n_sim))
            # draw.append('{:0.0f}%'.format(100*(hg==ag).sum()/n_sim))
            # way_win.append('{:0.0f}%'.format(100*(hg<ag).sum()/n_sim))
            home_win.append(100 * (hg > ag).sum() / n_sim)
            draw.append(100 * (hg == ag).sum() / n_sim)
            away_win.append(100 * (hg < ag).sum() / n_sim)
            average_home.append(hg.mean())
            average_away.append(ag.mean())

        f = 1
        home_win = np.round(home_win, f)
        draw = np.round(draw, f)
        away_win = np.round(away_win, f)
        average_home = np.round(average_home, decimals=f)
        average_away = np.round(average_away, decimals=f)
        df = pd.DataFrame(
            {'Date': date, 'Home': home, 'Away': away, 'Home Wins': home_win, 'Draw': draw, 'Away Wins': away_win,
             'av HG': average_home, 'av AG': average_away})
        return df.sort_values(by='Date')

    def simulate_season(self, n_scenarios=10000):
        nr_matches_to_sim = len(self.matches_to_sim)
        self.simulated_home_goals = np.zeros([nr_matches_to_sim, n_scenarios])
        self.simulated_away_goals = np.zeros([nr_matches_to_sim, n_scenarios])
        for i, match in enumerate(self.matches_to_sim):
            self.matches_to_sim[match]['id'] = i
            home_team = self.teams[self.matches_to_sim[match]['Home']]
            away_team = self.teams[self.matches_to_sim[match]['Away']]
            gH, gA, _ = home_team.vs(away_team, n=n_scenarios, home_advantage=self.home_advantage)
            self.simulated_home_goals[i, :] = gH
            self.simulated_away_goals[i, :] = gA
        self.simulation_done = True
        self.simulation_processed = False

    def what_if(self, match, ref_team, show_plot=True, place=4, or_better=True):
        if not self.simulation_done:
            print('simulation not yet done, simulating')
            self.simulate_season()
        if not self.simulation_processed:
            print('simulation not yet processed, processing')
            self.process_simulation()

        match_id = match['id']
        _home = match['Home']
        _away = match['Away']
        ref_team_name = ref_team.name
        home_goals = self.simulated_home_goals[match_id, :]
        away_goals = self.simulated_away_goals[match_id, :]
        home_won = home_goals > away_goals
        away_won = home_goals < away_goals
        draw = home_goals == away_goals
        ref_team_id = self.team_id[ref_team_name]
        place_if_home = self.place_per_team[ref_team_id, home_won]
        place_if_away = self.place_per_team[ref_team_id, away_won]
        place_if_draw = self.place_per_team[ref_team_id, draw]
        p_cl = np.zeros(4)
        if or_better:
            p_cl[0] = 100 * (self.place_per_team[ref_team_id] <= place).sum() / self.place_per_team[ref_team_id].shape[0]
            p_cl[1] = 100 * (place_if_home <= place).sum() / place_if_home.shape[0]
            p_cl[2] = 100 * (place_if_away <= place).sum() / place_if_away.shape[0]
            p_cl[3] = 100 * (place_if_draw <= place).sum() / place_if_draw.shape[0]
        else:
            p_cl[0] = 100 * (self.place_per_team[ref_team_id] == place).sum() / self.place_per_team[ref_team_id].shape[0]
            p_cl[1] = 100 * (place_if_home == place).sum() / place_if_home.shape[0]
            p_cl[2] = 100 * (place_if_away == place).sum() / place_if_away.shape[0]
            p_cl[3] = 100 * (place_if_draw == place).sum() / place_if_draw.shape[0]

        if show_plot:
            fig, ax = plt.subplots(1, 1)
            _width = 0.2
            x, y = p_plot(self.place_per_team[ref_team_id])
            xx = np.zeros(x.shape[0] + 1)
            yy = np.zeros(y.shape[0] + 1)
            x_cl = x[-1] + 1
            xx[:-1] = x
            xx[-1] = x_cl
            yy[:-1] = y
            yy[-1] = p_cl[0]
            xx0 = np.array(xx)

            ax.bar(xx - 1.5 * _width, yy, width=_width, label='Current. CL: {:0.2f}'.format(p_cl[0]))

            x, y = p_plot(place_if_home)
            xx = np.zeros(x.shape[0] + 1)
            yy = np.zeros(y.shape[0] + 1)
            xx[:-1] = x
            xx[-1] = x_cl
            yy[:-1] = y
            yy[-1] = p_cl[1]
            ax.bar(xx - 0.5 * _width, yy, width=_width, label='{:s} Win. CL: {:0.2f}'.format(_home, p_cl[1]))

            x, y = p_plot(place_if_away)
            xx = np.zeros(x.shape[0] + 1)
            yy = np.zeros(y.shape[0] + 1)
            xx[:-1] = x
            xx[-1] = x_cl
            yy[:-1] = y
            yy[-1] = p_cl[2]
            ax.bar(xx + 0.5 * _width, yy, width=_width, label='{:s} Win. CL: {:0.2f}'.format(_away, p_cl[2]))

            x, y = p_plot(place_if_draw)
            xx = np.zeros(x.shape[0] + 1)
            yy = np.zeros(y.shape[0] + 1)
            xx[:-1] = x
            xx[-1] = x_cl
            yy[:-1] = y
            yy[-1] = p_cl[3]
            ax.bar(xx + 1.5 * _width, yy, width=_width, label='Draw. CL: {:0.2f}'.format(p_cl[3]))
            ax.grid(True)
            _label = []
            for _x in xx0:
                _label.append(str(int(_x)))
            _label[-1] = 'CL'
            ax.set_xticks(xx0)
            ax.set_xticklabels(_label)
            ax.legend()
            ax.set_title(ref_team_name)
            fig.set_size_inches(16, 9)
            return p_cl, fig
        return p_cl, None

    def importance(self, match, ref_team, place=4, or_better=True):
        p = self.what_if(match, ref_team, show_plot=False, place=place, or_better=or_better)[0]
        return np.sum((p[1:]-p[0])**2)

    def process_simulation(self, max_date_index=np.inf, verbose=False):
        if not self.simulation_done:
            print('simulation not yet done, simulating')
            self.simulate_season()
        n_scenarios = self.simulated_home_goals.shape[1]
        points_per_team = np.zeros([self.nr_teams, n_scenarios])
        place_per_team = np.zeros([self.nr_teams, n_scenarios])
        goals_per_team = np.zeros([self.nr_teams, n_scenarios])
        goals_against_per_team = np.zeros([self.nr_teams, n_scenarios])

        for _team in self.teams:
            team_id = self.team_id[_team]
            points_per_team[team_id, :] = self.current_points[_team]
            goals_per_team[team_id, :] = self.current_goals[_team]
            goals_against_per_team[team_id, :] = self.current_goals_against[_team]

        for _match in self.matches_to_sim:
            _details = self.matches_to_sim[_match]
            _date = _details['Date']
            _date_index = _details['eff_date_index']
            if _date_index <= max_date_index:
                if verbose:
                    print(_date,_match)
                _home = _details['Home']
                _home_id = self.team_id[_home]
                _away = _details['Away']
                match_id = _details['id']
                _away_id = self.team_id[_away]
                home_goals = self.simulated_home_goals[match_id]
                away_goals = self.simulated_away_goals[match_id]
                goals_per_team[_home_id, :] += home_goals
                goals_per_team[_away_id, :] += away_goals
                goals_against_per_team[_home_id, :] += away_goals
                goals_against_per_team[_away_id, :] += home_goals
                home_won = home_goals > away_goals
                away_won = home_goals < away_goals
                draw = home_goals == away_goals
                points_per_team[_home_id, home_won] += 3
                points_per_team[_home_id, draw] += 1
                points_per_team[_away_id, away_won] += 3
                points_per_team[_away_id, draw] += 1
            else:
                trash = 1

        modified_points = np.zeros([self.nr_teams, n_scenarios])
        modified_points += points_per_team
        b = (goals_per_team - goals_against_per_team).max(axis=0)
        a = (goals_per_team - goals_against_per_team).min(axis=0)
        modified_points += 0.1 * ((goals_per_team - goals_against_per_team) - a) / (b - a)
        b = goals_per_team.max(axis=0)
        a = goals_per_team.min(axis=0)
        modified_points += 0.01 * (goals_per_team - a) / (b - a)
        modified_points += 0.001 * np.random.random(modified_points.shape)
        place_per_team = 0 * modified_points
        ordering = (-modified_points).argsort(axis=0)
        for _team_id in range(self.nr_teams):
            a, b = np.where(ordering == _team_id)
            place_per_team[_team_id, b] = a + 1

        self.place_per_team = place_per_team
        self.points_per_team = points_per_team
        self.goals_per_team = goals_per_team
        self.goals_against_per_team = goals_against_per_team
        self.simulation_processed = True

    def season_report(self, ind=None):
        if not self.simulation_done:
            print('simulation not yet done, simulating')
            self.simulate_season()
        if not self.simulation_processed:
            print('simulation not yet processed, processing')
            self.process_simulation()
        if ind is None:
            ind = np.ones(self.points_per_team.shape[1]).astype(bool)
        average_points = self.points_per_team[:, ind].mean(axis=1).round(1)
        average_goals = self.goals_per_team[:, ind].mean(axis=1).round(1)
        average_goals_against = self.goals_against_per_team[:, ind].mean(axis=1).round(1)
        p_win = (100 * (self.place_per_team[:, ind] == 1).sum(axis=1) / ind.sum()).round(2)
        p_cl = (100 * (self.place_per_team[:, ind] <= self.nr_cl).sum(axis=1) / ind.sum()).round(2)
        p_degr = (100 * (self.place_per_team[:, ind] > self.nr_teams - self.nr_degr).sum(axis=1) / ind.sum()).round(2)
        points_up = np.percentile(self.points_per_team[:, ind], 95, axis=1).round(0)
        points_down = np.percentile(self.points_per_team[:, ind], 5, axis=1).round(0)
        place_up = np.percentile(self.place_per_team[:, ind], 5, axis=1).round(0)
        place_down = np.percentile(self.place_per_team[:, ind], 95, axis=1).round(0)
        team_names = []
        lmbd = []
        tau = []
        current_points = []
        played = []
        replace_dict = {'\xfc': 'ue', '\xe9': 'e'}
        for _i in self.inv_team_id:
            team_name = self.inv_team_id[_i]
            team_name_ascii = team_name
            for _from, _to in replace_dict.items():
                team_name_ascii = team_name_ascii.replace(_from, _to)
            current_points.append(self.current_points[team_name])
            played.append(self.played[team_name])
            team_names.append(team_name_ascii)
            _l, _t = self.teams[team_name].means()
            lmbd.append(_l)
            tau.append(_t)
        tau = np.array(tau).round(2)
        lmbd = np.array(lmbd).round(2)

        df = pd.DataFrame({'Played': played,
                           'Points (current)': current_points,
                           'Points (mean)': average_points,
                           'Points (high)': points_up.astype(int),
                           'Points (low)': points_down.astype(int),
                           'Place (high)': place_up.astype(int),
                           'Place (low)': place_down.astype(int),
                           'GF': average_goals,
                           'GA': average_goals_against,
                           'GD': average_goals - average_goals_against,
                           'CL': p_cl,
                           'Win': p_win,
                           'Degr': p_degr,
                           'Off': lmbd,
                           'Deff': tau},
                          index=team_names)
        df = df.sort_values(by='Points (mean)', ascending=False)
        cols = ['Played','Points (current)', 'Points (mean)', 'Points (low)', 'Points (high)', 'Place (low)', 'Place (high)', 'Win', 'CL', 'Off',
                'Deff', 'Degr']
        return df[cols]

    def points_probability_grid(self):
        x0 = self.points_per_team.min() - 1
        x1 = self.points_per_team.max() + 3
        step = (x1 - x0) / (1 + 1.5 * self.nr_teams)
        step = np.floor(step).astype(int)
        bins = np.arange(x0, x1, step).astype(int)
        # bins = np.unique(np.round(np.linspace(self.points_per_team.min()-1,self.points_per_team.max()+1,int(1+1.5*self.nr_teams))).astype(int))


        labels = ['[{:d},{:d})'.format(a, b) for a, b in zip(bins[0:-1], bins[1:])]

        T = np.zeros([self.nr_teams, bins.size-1])
        team_names = []
        for name, i in self.team_id.items():
            #T[i, :] = np.bincount(self.points_per_team[i, :].astype(int), minlength=int(b) + 1)
            T[i, :] = np.histogram(self.points_per_team[i, :], bins)[0]
            T[i, :] = 100 * T[i, :] / T[i, :].sum()
            team_names.append(name)
        #x = np.arange(int(b) + 1)

        ind = T.max(axis=0) > 0

        isort = np.argsort(self.points_per_team.mean(axis=1))
        T = T[isort, :]
        T = T[:, ind]
        # x = x[ind]
        team_names = np.array(team_names)[isort]
        fig = plt.figure(figsize=(ind.sum() * 0.7, self.nr_teams * 0.7))
        plt.imshow(T, cmap='binary')
        for i in range(self.nr_teams):
            plt.axhline(y=i - 0.5, color='k', linewidth=1)
            for j in range(ind.sum()):
                if T[i, j] >= 50:
                    color = 'white'
                else:
                    color = 'green'
                if T[i, j] > 0:
                    plt.text(j - 0.35, i + 0.1, '{:0.1f}%'.format(T[i, j]), color=color)
        for i in range(ind.sum()):
            plt.axvline(x=i - 0.5, color='k', linewidth=1)
        plt.ylim([-0.5, self.nr_teams - 0.5])
        current_points = np.array([self.current_points[x] for x in team_names])
        # y = [[i for i, a in enumerate(zip(bins[0:-1], bins[1:])) if a[0] <= x < a[1]][0] for x in current_points]
        y = [[(i + (x - a[0]) / (a[1] - a[0]) - 0.5) for i, a in enumerate(zip(bins[0:-1], bins[1:])) if
              a[0] <= x < a[1]][0] for x in current_points]
        plt.plot(y, np.arange(self.nr_teams), 'r.')
        #plt.xticks(np.arange(ind.sum()), x, rotation='vertical')
        plt.xticks(np.arange(len(labels)), labels, rotation='vertical');
        plt.yticks(np.arange(len(team_names)), team_names)
        # plt.grid(True)
        return team_names, T, fig

    def probability_grid(self, ind=None, title=''):
        if not self.simulation_done:
            print('simulation not yet done, simulating')
            self.simulate_season()
        if not self.simulation_processed:
            print('simulation not yet processed, processing')
            self.process_simulation()
        if ind is None:
            ind = np.ones(self.points_per_team.shape[1]).astype(bool)

        fig = plt.figure(1)

        T = np.zeros([self.nr_teams, self.nr_teams])
        team_names = ['']*self.nr_teams
        current_points = np.zeros(self.nr_teams)
        current_goals = np.zeros(self.nr_teams)
        current_goals_against = np.zeros(self.nr_teams)
        for name, i in self.team_id.items():
            T[i, :] = np.bincount(self.place_per_team[i, ind].astype(int) - 1, minlength=self.nr_teams)
            T[i, :] = 100 * T[i, :] / T[i, :].sum()
            team_names[i]=name
            current_points[i] = self.current_points[name]
            current_goals[i] = self.current_goals[name]
            current_goals_against[i] = self.current_goals_against[name]
        modified_points = np.zeros(self.nr_teams)
        modified_points += current_points
        b = (current_goals - current_goals_against).max()
        a = (current_goals - current_goals_against).min()
        modified_points += 0.1 * ((current_goals - current_goals_against) - a) / (b - a)
        b = current_goals.max()
        a = current_goals.min()
        modified_points += 0.01 * (current_goals - a) / (b - a)
        modified_points += 0.001 * np.random.random(modified_points.shape)
        # i_sort = (-self.points_per_team[:, ind].mean(axis=1)).argsort()
        i_sort = (-modified_points).argsort()
        team_names = np.array(team_names)[i_sort]
        yticks = ['{:s} ({:d}pt in {:d}) {:d}'.format(name,self.current_points[name],self.played[name],i+1) for i,name in enumerate(team_names)]
        T = T[i_sort, :]
        plt.imshow(T, cmap='binary')
        for i in range(self.nr_teams):
            if (i == self.nr_cl) | (i == self.nr_teams - self.nr_degr):
                plt.axvline(x=i - 0.5, color='r', linewidth=1)
                plt.axhline(y=i - 0.5, color='r', linewidth=1)
            else:
                plt.axvline(x=i - 0.5, color='k', linewidth=1)
                plt.axhline(y=i - 0.5, color='k', linewidth=1)


            for j in range(self.nr_teams):
                if T[i, j] >= 50:
                    color = 'white'
                else:
                    color = 'green'
                if T[i, j] >= 1:
                    plt.text(j - 0.35, i + 0.1, '{:0.0f}%'.format(T[i, j]), color=color)
                elif T[i, j] >= 0.1:
                    plt.text(j - 0.35, i + 0.1, '{:0.1f}%'.format(T[i, j]), color=color)

        # plt.colorbar()
        plt.yticks(np.arange(self.nr_teams), yticks)
        plt.xticks(np.arange(self.nr_teams), np.arange(self.nr_teams) + 1)
        plt.plot(np.arange(self.nr_teams-1)+0.5,np.arange(self.nr_teams-1)+0.5,'+r')
        # plt.grid(True)
        plt.title(title)
        fig.set_size_inches(16, 12)
        return team_names,T,fig

    def team_report(self, team):
        if not self.simulation_done:
            print('simulation not yet done, simulating')
            self.simulate_season()
        if not self.simulation_processed:
            print('simulation not yet processed, processing')
            self.process_simulation()
        team_name = team.name
        fig, ax = plt.subplots(2, 2)
        x, y = p_plot(self.place_per_team[self.team_id[team_name], :])
        ax[0, 0].bar(x, y)
        ax[0, 0].set_xticks(x)
        ax[0, 0].set_title('Place')
        x, y = p_plot(self.points_per_team[self.team_id[team_name], :])
        ax[0, 1].bar(x, y)
        ax[0, 1].bar(self.current_points[team_name], y.max())
        ax[0, 1].set_title('Points')
        x, y = p_plot(
            self.goals_per_team[self.team_id[team_name], :] - self.goals_against_per_team[self.team_id[team_name], :])
        ax[1, 0].bar(x, y)
        ax[1, 0].bar(self.current_goals[team_name] - self.current_goals_against[team_name], y.max())
        ax[1, 0].set_title('Goal Difference')
        x, y = p_plot(self.goals_per_team[self.team_id[team_name], :])
        ax[1, 1].bar(x, y)
        ax[1, 1].bar(self.current_goals[team_name], y.max())
        ax[1, 1].set_title('Goals')

        for _i in ax:
            for _j in _i:
                _j.grid(True)

        fig.set_size_inches(16, 9)

