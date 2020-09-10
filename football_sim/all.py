import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
import yaml
from copy import deepcopy
import os
import pickle


class Settings:
    def __init__(self, settings_file):
        self.league_info = yaml.safe_load(open(settings_file, 'r'))
        self.domestic_leagues = list(self.league_info.keys())
        self.eu_leagues = ['UCL', 'UEL']

    def get_info(self, name):
        if name in self.league_info:
            return self.league_info[name]


class Calibrator:
    def __init__(self, settings, sigma_x=0.45, sigma_y=0.5):
        self.raw_data = None
        self.fixtures = []
        self.teams = dict()
        self.settings = settings
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.file_name = 'calibrator.pkl'
        self.initialized=[]

    def initialize_calibration(self, league, year):
        completed_fixtures = self.get_completed_fixtures(league, year - 1)
        dates = [f.date for f in completed_fixtures]
        is_sorted = np.all(dates[:-1] <= dates[1:])
        assert is_sorted
        [f.update_teams(update_forecast=True) for f in completed_fixtures]
        for f in completed_fixtures:
            f.home_team.forget_history()
            f.away_team.forget_history()
            f.league_home_team.forget_history()
            f.league_away_team.forget_history()
        self.initialized.append([league, year])

    def calibrate_teams(self, league, year, as_of=None):
        if [league,year] not in self.initialized:
            self.initialize_calibration(league,year)
        completed_fixtures = self.get_completed_fixtures(league, year, as_of=as_of)
        dates = [f.date for f in completed_fixtures]
        is_sorted = np.all(dates[:-1] <= dates[1:])
        assert is_sorted
        [f.update_teams(update_forecast=True) for f in completed_fixtures]

    def reset_teams(self):
        for team in self.teams.values():
            team.reset()

    def get_completed_fixtures(self, league_name, year, as_of=None):
        all_fixtures = self.get_fixtures_for_league(league_name, year)
        if as_of is None:
            return [f for f in all_fixtures if f.completed]
        else:
            return [f for f in all_fixtures if f.completed and f.date <= as_of]

    def get_remaining_fixtures(self, league_name, year, as_of=None):
        all_fixtures = self.get_fixtures_for_league(league_name, year)
        completed_fixtures = self.get_completed_fixtures(league_name, year, as_of=as_of)
        return [f for f in all_fixtures if f not in completed_fixtures]

    def get_fixtures_for_league(self, league_name, year):
        return [f for f in self.fixtures if f.year == year and f.league == league_name]

    def get_teams_for_league(self, league_name, year):
        fixtures = self.get_fixtures_for_league(league_name, year)
        team_id = [f.home_team.id for f in fixtures] + [f.away_team.id for f in fixtures]
        team_id = np.unique(team_id)
        return {self.teams[x].name: self.teams[x] for x in team_id}

    def process_data(self):
        ind = self.raw_data['League'].apply(lambda x: x in self.settings.domestic_leagues)
        for i, row in self.raw_data.loc[ind].sort_values(by='Date').iterrows():
            f = Fixture(row)
            f.set_teams(self)
            self.fixtures.append(f)

    def get_team(self, league, name):
        if 'Home' in name or 'Away' in name:
            team_ = Team(name=name, league=league, sigma_x=self.sigma_x / 2, sigma_y=self.sigma_y / 2)
        else:
            team_ = Team(name=name, league=league, sigma_x=self.sigma_x, sigma_y=self.sigma_y)
        if team_.id not in self.teams:
            self.teams[team_.id] = team_
        return self.teams[team_.id]

    def download_all_data(self):
        f = lambda x: "".join([y[0] for y in x.upper().split()])
        df = pd.read_csv('https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv')
        df = df.rename(columns={'team1': 'HomeTeam', 'team2': 'AwayTeam', 'score1': 'FTHG', 'score2': 'FTAG',
                                'league': 'League', 'season': 'Season'})
        df['Date'] = pd.to_datetime(df['date'])
        df = df[['Date', 'Season', 'League', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'xg1', 'xg2', 'nsxg1',
                 'nsxg2', 'adj_score1', 'adj_score2']]
        df['League'] = df['League'].apply(f)
        self.raw_data = df

    def copy(self):
        return deepcopy(self)


class Fixture:
    def __init__(self, row):
        self.league = row['League']
        self.home_team_name = row['HomeTeam']
        self.away_team_name = row['AwayTeam']
        self.year = row['Season']
        self.league_home_team = None
        self.league_away_team = None
        self.home_team = None
        self.away_team = None
        self.date = row['Date']
        self.home_goals = row['FTHG']
        self.away_goals = row['FTAG']
        if ~np.isnan(self.home_goals):
            if self.home_goals > self.away_goals:
                self.result = 0
            elif self.home_goals < self.away_goals:
                self.result = 1
            else:
                self.result = 2
        self.home_metrics = [row['xg1'], row['nsxg1'], row['adj_score1']]
        self.away_metrics = [row['xg2'], row['nsxg2'], row['adj_score2']]
        if np.isnan(self.home_metrics).any() or np.isnan(self.away_metrics).any():
            self.home_ag = self.home_goals
            self.away_ag = self.away_goals
        else:
            self.home_ag = np.mean(self.home_metrics)
            self.away_ag = np.mean(self.away_metrics)
        # self.home_ag = self.home_goals
        # self.away_ag = self.away_goals

        if np.isnan(self.home_ag) or np.isnan(self.away_ag):
            self.completed = False
        else:
            self.completed = True

        self.id = (
            '_'.join([self.league, self.home_team_name, self.away_team_name, self.date.strftime('%Y-%m-%d')])).replace(
            ' ', '').lower()
        self.forecast_home_goals = None
        self.forecast_away_goals = None
        self.forecast_home_wins = None
        self.forecast_away_wins = None
        self.forecast_draw = None
        self.forecast_result = None
        self.forecast_probability = None
        self.used_for_calibrating = False

    def set_teams(self, calibrator):
        self.home_team = calibrator.get_team(self.league, self.home_team_name)
        self.away_team = calibrator.get_team(self.league, self.away_team_name)
        self.home_team.add_fixture(self)
        self.away_team.add_fixture(self)
        self.league_home_team = calibrator.get_team(self.league + '_', self.league + 'Home')
        self.league_away_team = calibrator.get_team(self.league + '_', self.league + 'Away')

    def simulate(self, n=int(1e4), home_advantage=1):
        gh, ga, des = self.home_team.vs(self.away_team, n=n, home_advantage=home_advantage)
        self.forecast_home_goals = gh.mean()
        self.forecast_away_goals = ga.mean()
        self.forecast_home_wins = 100 * np.sum(gh > ga) / n
        self.forecast_away_wins = 100 * np.sum(gh < ga) / n
        self.forecast_draw = 100 * np.sum(gh == ga) / n
        self.forecast_result = np.argmax([self.forecast_home_wins, self.forecast_away_wins, self.forecast_draw])
        self.forecast_probability = np.max([self.forecast_home_wins, self.forecast_away_wins, self.forecast_draw])
        return gh, ga, des

    def update_teams(self, update_forecast=False):
        if (~np.isnan(self.home_ag)) & (~np.isnan(self.away_ag)) & (~self.used_for_calibrating):
            if update_forecast:
                self.simulate()

            self.home_team.scored_against(self.away_team, self.home_ag)
            self.away_team.scored_against(self.home_team, self.away_ag)
            self.home_team.t.append(self.date)
            self.away_team.t.append(self.date)
            self.league_home_team.t.append(self.date)
            self.league_away_team.t.append(self.date)
            self.league_home_team.scored_against(self.league_away_team, self.home_ag)
            self.league_away_team.scored_against(self.league_home_team, self.away_ag)
            self.used_for_calibrating = True
        return self

    def __repr__(self):
        h = [self.home_goals] + self.home_metrics + [self.forecast_home_goals, self.forecast_home_wins, self.forecast_draw]
        a = [self.away_goals] + self.away_metrics + [self.forecast_away_goals, self.forecast_away_wins, self.forecast_draw]
        i = ['goals', 'xg', 'nsxg', 'ag', 'forecasted goals', 'win probability', 'draw probability']
        df = pd.DataFrame({'metric': i, self.home_team_name: h, self.away_team_name: a})
        return df.__repr__()


class Team(object):
    def __init__(self, name='team name', league='SH', sigma_x=0.5, sigma_y=0.5):
        self.name = name
        self.league = league
        self.id = '{:s}_{:s}'.format(league, name.replace(' ', ''))
        self.fixtures = []
        # self.id = '{:s}_{:s}'.format(league,name.replace(' ',''))
        self.lmbd_set = np.linspace(0, 10, 101)
        self.p = self.lmbd_set * 0 + 1
        self.p = self.p / self.p.sum()
        self.tau_set = np.linspace(0, 1, 101)
        self.q = self.tau_set * 0 + 1
        self.q = self.q / self.q.sum()
        self.lambda_fun = lambda x: 10 / (1 + np.exp(-np.array(x)))
        self.p_fun = lambda x: 1 / (1 + np.exp(-np.array(x)))
        self.x_hist = []
        self.y_hist = []
        self.x = 0
        self.y = 0
        self.t = []
        self.offense_hist = []
        self.defense_hist = []
        self.offense = self.lambda_fun(self.x)
        self.defense = self.p_fun(self.y)

        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

        self.bayesian = False

    def set_x(self, x):
        self.x = x
        self.x_hist.append(x)
        self.offense = self.lambda_fun(self.x)
        self.offense_hist.append(self.offense)

    def set_y(self, y):
        self.y = y
        self.y_hist.append(y)
        self.defense = self.p_fun(self.y)
        self.defense_hist.append(self.defense)

    def reset(self):
        self.x_hist = []
        self.y_hist = []
        self.x = 0
        self.y = 0
        self.t = []
        self.offense_hist = []
        self.defense_hist = []
        self.offense = self.lambda_fun(self.x)
        self.defense = self.p_fun(self.y)

    def forget_history(self):
        self.x_hist = []
        self.y_hist = []
        self.offense_hist = []
        self.defense_hist = []
        self.t = []

    def add_fixture(self, fixture):
        if fixture not in self.fixtures:
            self.fixtures.append(fixture)

    def __repr__(self):
        return self.name

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

    def outcomes_vs(self, other_team, n_scenarios=int(1e5), home_advantage=1, t=90, g_h=0, g_a=0):
        g = np.zeros([n_scenarios, 2])
        g[:, 0], g[:, 1], _ = self.vs(other_team, n=n_scenarios, home_advantage=home_advantage, t=t, g_h=g_h, g_a=g_a)
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
        plt.xticks(x, u, rotation='vertical')
        plt.legend()
        plt.xlim(-0.5, x[p > 0.5].max() + 0.5)
        plt.grid()
        gmean = g.mean(axis=0)
        plt.title('{} ({:.2f}) vs {} ({:.2f})'.format(self.name, gmean[0], other_team.name, gmean[1]))

    def vs(self, other_team, n=int(1e4), home_advantage=1, t=90, g_h=0, g_a=0):

        if self.bayesian:
            l_h = (t/90)*np.random.choice(self.lmbd_set, size=n, p=self.p) * np.random.choice(other_team.tau_set, size=n, p=other_team.q) * home_advantage
            l_a = (t/90)*np.random.choice(other_team.lmbd_set, size=n, p=other_team.p) * np.random.choice(self.tau_set, size=n, p=self.q)
            g_h += np.random.poisson(l_h)
            g_a += np.random.poisson(l_a)
        else:

            l_h = (t/90)*self.offense * other_team.defense * home_advantage
            l_a = (t/90)*other_team.offense * self.defense
            g_h += np.random.poisson(l_h, n)
            g_a += np.random.poisson(l_a, n)

        match_des = self.name + ' vs ' + other_team.name
        return g_h, g_a, match_des

    def plt(self, ax=None, colors=None):
        if ax is None:

            if self.bayesian:
                fig, ax = plt.subplots(1, 2)
            else:
                fig, ax = plt.subplots(2, 1)
            fig.set_size_inches(16, 9)
        l, t = self.means()
        if self.bayesian:
            p1 = ax[0].plot(self.lmbd_set, self.p, label=self.name + ' off: {:0.2f}'.format(l))
            ax[1].plot(self.tau_set, self.q, c=p1[0].get_color(), label=self.name + ' def: {:0.2f}'.format(t))
        else:
            ut = self.t
            it = np.array([(x - self.t[0]) / pd.Timedelta('1 day') for x in self.t])
            # it = np.arange(len(self.t))
            if colors == None:
                p1 = ax[0].plot(it, self.offense_hist, label=self.name + ' off: {:0.2f}'.format(l))
                colors = [p1[0].get_color(), p1[0].get_color()]
            else:
                p1 = ax[0].plot(it, self.offense_hist, label=self.name + ' off: {:0.2f}'.format(l), color=colors[0])
            xticks = np.linspace(0, it.max(), 20)
            labels = [ut[0] + (ut[-1] - ut[0]) * x for x in xticks / xticks.max()]
            ax[0].set_xticks(xticks)
            ax[0].set_xticklabels([x.strftime('%Y-%m-%d') for x in labels], rotation=90)
            ax[1].plot(it, self.defense_hist, c=colors[1], label=self.name + ' def: {:0.2f}'.format(t))
            ax[1].set_xticks(xticks)
            ax[1].set_xticklabels([x.strftime('%Y-%m-%d') for x in labels], rotation=90)


        ax[0].legend()
        ax[0].grid(True)
        ax[1].legend()
        ax[1].grid(True)
        # l, t = self.means()
        # ax[0].set_title('lambda: {:0.2f}'.format(l))
        # ax[1].set_title('tau: {:0.2f}'.format(t))
        return ax

    def means(self):
        if self.bayesian:
            return self.p.dot(self.lmbd_set), self.q.dot(self.tau_set)
        else:
            return self.offense, self.defense

    def target_fun(self, x, other, k):
        lmbd = self.lambda_fun(x[0])
        p = other.p_fun(x[1])
        lp = lmbd * p
        x0 = (x[0] - self.x) / self.sigma_x
        y0 = (x[1] - other.y) / other.sigma_y
        return -(lp ** k) * np.exp(-lp) * norm.pdf(x0) * norm.pdf(y0)

    def scored_against(self, other, k):
        lmb_times_tau = self.lmbd_set * other.tau_set[:, np.newaxis]
        new_p = ((np.exp(-lmb_times_tau) * (lmb_times_tau ** k)).T * other.q).sum(axis=1) * self.p
        self.p = new_p / new_p.sum()
        new_q = ((np.exp(-lmb_times_tau) * (lmb_times_tau ** k)) * self.p).sum(axis=1) * other.q
        other.q = new_q / new_q.sum()

        sol = minimize(self.target_fun, np.array([self.x, other.y]), args=(other, k))
        self.set_x(sol.x[0])
        other.set_y(sol.x[1])


def p_plot(x):
    a = x.min()
    b = x.max()
    xx = np.arange(a, b + 1)
    yy = xx * 0
    for _i in range(xx.shape[0]):
        yy[_i] = np.sum(x == xx[_i])
    yy = 100 * yy / yy.sum()
    return xx, yy


class Season:
    def __init__(self, name, year, calibrator, use_home_advantage=True, base_folder='./',as_of = None):
        self.as_of = as_of
        self.output_folder = os.path.join(base_folder, name)
        self.today = pd.to_datetime('today')
        self.today_str = self.today.strftime('%Y-%m-%d')
        if not os.path.isdir(self.output_folder):
            os.mkdir(self.output_folder)
        self.name = name
        self.year = year
        self.calibrator = calibrator
        self.teams = calibrator.get_teams_for_league(name, year)
        if use_home_advantage:
            league_home_team = calibrator.get_team(self.name + '_', self.name + 'Home')
            league_away_team = calibrator.get_team(self.name + '_', self.name + 'Away')
            l_h, p_h = league_home_team.means()
            l_a, p_a = league_away_team.means()
            self.home_advantage = l_h * p_a / (l_a * p_h)
        else:
            self.home_advantage = 1
        info = calibrator.settings.get_info(self.name)
        self.nr_cl = info['nr_cl']
        self.nr_degr = info['nr_deg']
        self.nr_teams = len(self.teams)
        self.all_matches = None
        self.matches_to_sim = None
        self.current_goals = dict()
        self.current_goals_against = dict()
        self.current_points = dict()
        self.played = dict()
        self.simulated_home_goals = None
        self.simulated_away_goals = None
        self.simulated_home_points = None
        self.simulated_away_points = None
        self.team_id = dict()
        self.inv_team_id = dict()
        i = 0
        self.current_modified_points = dict()
        self.current_place_per_team = dict()

        for _team in self.teams:
            self.current_modified_points[_team] = 0
            self.team_id[_team] = i
            self.inv_team_id[i] = _team
            self.current_goals[_team] = 0
            self.current_goals_against[_team] = 0
            self.current_points[_team] = 0
            self.played[_team] = 0
            i += 1
        self.simulation_done = False
        self.simulation_processed = False
        self.match_id = None
        self.league_start = None

    def process_current_results(self):
        completed_fixtures = self.calibrator.get_completed_fixtures(self.name, self.year, as_of=self.as_of)
        if len(completed_fixtures)>0:
            self.league_start = np.min([f.date for f in completed_fixtures])
        else:
            self.league_start = pd.to_datetime('today')
        for f in completed_fixtures:
            home_goals = f.home_goals
            away_goals = f.away_goals
            home_team = f.home_team.name
            away_team = f.away_team.name

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

        gd = [self.current_goals[x] - self.current_goals_against[x] for x in self.current_goals]
        a = np.min(gd)
        b = np.max(gd)
        g = [self.current_goals[x]  for x in self.current_goals]
        c = np.min(g)
        d = np.max(g)
        for team_name in self.current_points:
            self.current_modified_points[team_name] += self.current_points[team_name]
            g_ = self.current_goals[team_name]
            gd_ = g_ - self.current_goals_against[team_name]
            self.current_modified_points[team_name] += 0.1 * (gd_ - a) / (b - a)
            self.current_modified_points[team_name] += 0.01 * (g_ - c) / (d - c)
            self.current_modified_points[team_name] += 0.001 * np.random.random()

        team_names = []
        mp = []
        for team_name,p in self.current_modified_points.items():
            team_names.append(team_name)
            mp.append(p)
        i_sort = np.argsort(-np.array(mp))
        sorted_teams = np.array(team_names)[i_sort]

        for i,team_name in enumerate(sorted_teams):
            self.current_place_per_team[team_name] = i+1

        self.matches_to_sim = self.calibrator.get_remaining_fixtures(self.name, self.year, as_of=self.as_of)
        self.match_id = {match.id: i for i, match in enumerate(self.matches_to_sim)}

    def matches_remaining(self, team_filter=('',)):

        home = []
        away = []
        date = []
        home_win = []
        draw = []
        away_win = []
        average_home = []
        average_away = []
        n_sim = self.simulated_away_goals.shape[1]
        for i, match in enumerate(self.matches_to_sim):
            home_ = match.home_team.name
            away_ = match.away_team.name
            if np.any([(f_ in home_) or (f_ in away_) for f_ in team_filter]):
                home.append(home_)
                away.append(away_)
                date_ = match.date
                date.append(date_)
                hg = self.simulated_home_goals[i, :]
                ag = self.simulated_away_goals[i, :]
                home_win.append(100 * np.sum(hg > ag) / n_sim)
                draw.append(100 * np.sum(hg == ag) / n_sim)
                away_win.append(100 * np.sum(hg < ag) / n_sim)
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
            g_h, g_a, _ = match.simulate(n=n_scenarios, home_advantage=self.home_advantage)
            self.simulated_home_goals[i, :] = g_h
            self.simulated_away_goals[i, :] = g_a
        self.simulation_done = True
        self.simulation_processed = False

    def what_if(self, match, ref_team, show_plot=True, place=4, or_better=True):
        if not self.simulation_done:
            print('simulation not yet done, simulating')
            self.simulate_season()
        if not self.simulation_processed:
            print('simulation not yet processed, processing')
            self.process_simulation()

        match_id = self.match_id[match.id]
        _home = match.home_team.name
        _away = match.away_team.name
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
            p_cl[0] = 100 * (self.place_per_team[ref_team_id] <= place).sum() / self.place_per_team[ref_team_id].shape[
                0]
            p_cl[1] = 100 * (place_if_home <= place).sum() / place_if_home.shape[0]
            p_cl[2] = 100 * (place_if_away <= place).sum() / place_if_away.shape[0]
            p_cl[3] = 100 * (place_if_draw <= place).sum() / place_if_draw.shape[0]
        else:
            p_cl[0] = 100 * (self.place_per_team[ref_team_id] == place).sum() / self.place_per_team[ref_team_id].shape[
                0]
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

    def process_simulation(self):
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

        for i, match in enumerate(self.matches_to_sim):
            _date = match.date

            _home = match.home_team.name
            _home_id = self.team_id[_home]
            _away = match.away_team.name
            match_id = i
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

    def season_report(self, ind=None, file_name=None):
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
        strength = np.array(lmbd)/np.array(tau)
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
                           'Deff': tau,
                           'rating':strength},
                          index=team_names)
        df = df.sort_values(by='Points (mean)', ascending=False)
        cols = ['Played', 'Points (current)', 'Points (mean)', 'Points (low)', 'Points (high)', 'Place (low)',
                'Place (high)', 'Win', 'CL', 'Off',
                'Deff', 'rating' , 'Degr']
        if file_name is not None:
            if self.as_of is None:
                file_name = file_name + '_' + self.today_str
            else:
                file_name = file_name + '_' + self.as_of.strftime('%Y-%m-%d')
            file_name = os.path.join(self.output_folder, file_name+'.html')
            df[cols].to_html(file_name)

        return df[cols]

    def points_probability_grid(self, ind=None, file_name=None):
        if ind is None:
            ind = np.ones(self.points_per_team.shape[1]).astype(bool)
        x0 = self.points_per_team.min() - 1
        x1 = self.points_per_team.max() + 3
        step = (x1 - x0) / (1 + 1.5 * self.nr_teams)
        step = np.floor(step).astype(int)
        bins = np.arange(x0, x1, step).astype(int)
        # bins = np.unique(np.round(np.linspace(self.points_per_team.min()-1,self.points_per_team.max()+1,int(1+1.5*self.nr_teams))).astype(int))
        labels = ['[{:d},{:d})'.format(a, b) for a, b in zip(bins[0:-1], bins[1:])]


        T = np.zeros([self.nr_teams, bins.size - 1])
        team_names = []
        for name, i in self.team_id.items():
            # T[i, :] = np.bincount(self.points_per_team[i, :].astype(int), minlength=int(b) + 1)
            T[i, :] = np.histogram(self.points_per_team[i, ind], bins)[0]
            T[i, :] = 100 * T[i, :] / T[i, :].sum()
            team_names.append(name)
        # x = np.arange(int(b) + 1)

        ind = T.max(axis=0) >= 0

        isort = np.argsort(self.points_per_team.mean(axis=1))
        T = T[isort, :]
        T = T[:, ind]
        # x = x[ind]
        team_names = np.array(team_names)[isort]
        fig = plt.figure(figsize=(ind.sum() * 0.7, self.nr_teams * 0.7))
        fig.clear()
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
        # y = [[(i + (x - a[0]) / (a[1] - a[0]) - 0.5) for i, a in enumerate(zip(bins[0:-1], bins[1:])) if  a[0] <= x < a[1]][0] for x in current_points]
        f_points = lambda x: -0.5 + (x - bins.min()) / (bins.max() - bins.min()) * ind.sum()
        y = f_points(current_points)
        plt.plot(y, np.arange(self.nr_teams), 'r.')
        # plt.xticks(np.arange(ind.sum()), x, rotation='vertical')
        plt.xticks(np.arange(len(labels)), labels, rotation='vertical');
        labels_ = ['{:s} ({:0.0f})'.format(nm, pnt) for nm, pnt in zip(team_names, current_points)]
        plt.yticks(np.arange(len(team_names)), labels_)
        # plt.grid(True)
        if file_name is not None:
            if self.as_of is None:
                file_name = file_name + '_' + self.today_str
            else:
                file_name = file_name + '_' + self.as_of.strftime('%Y-%m-%d')
            file_name = os.path.join(self.output_folder, file_name+'.png')
            fig.savefig(file_name)
        return team_names, T, fig

    def probability_grid(self, ind=None, title='', file_name=None):
        if not self.simulation_done:
            print('simulation not yet done, simulating')
            self.simulate_season()
        if not self.simulation_processed:
            print('simulation not yet processed, processing')
            self.process_simulation()
        if ind is None:
            ind = np.ones(self.points_per_team.shape[1]).astype(bool)

        fig = plt.figure(1)
        fig.clear()

        T = np.zeros([self.nr_teams, self.nr_teams])
        team_names = [''] * self.nr_teams
        current_points = np.zeros(self.nr_teams)
        current_goals = np.zeros(self.nr_teams)
        current_goals_against = np.zeros(self.nr_teams)
        for name, i in self.team_id.items():
            T[i, :] = np.bincount(self.place_per_team[i, ind].astype(int) - 1, minlength=self.nr_teams)
            T[i, :] = 100 * T[i, :] / T[i, :].sum()
            team_names[i] = name
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
        yticks = ['{:s} ({:d}pt in {:d}) {:d}'.format(name, self.current_points[name], self.played[name], i + 1) for
                  i, name in enumerate(team_names)]
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
        plt.plot(np.arange(self.nr_teams - 1) + 0.5, np.arange(self.nr_teams - 1) + 0.5, '+r')
        # plt.grid(True)
        plt.title(title)
        fig.set_size_inches(16, 12)
        if file_name is not None:
            if self.as_of is None:
                file_name = file_name + '_' + self.today_str
            else:
                file_name = file_name + '_' + self.as_of.strftime('%Y-%m-%d')
            file_name = os.path.join(self.output_folder, file_name+'.png')
            fig.savefig(file_name)

        return team_names, T, fig

    def team_report(self, team, file_name=None, places=None):
        if places is None:
            places = [1, self.nr_cl, self.nr_teams - self.nr_degr]
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
        ax[0, 0].set_title('Place: {:0.0f}'.format(self.current_place_per_team[team_name]))
        ax[0, 0].grid(True)
        x, y = p_plot(
            self.goals_per_team[self.team_id[team_name], :] - self.goals_against_per_team[self.team_id[team_name], :])
        ax[1, 0].bar(x, y)
        # ax[1, 0].bar(self.current_goals[team_name] - self.current_goals_against[team_name], y.max())
        gd = self.current_goals[team_name]-self.current_goals_against[team_name]
        ax[1, 0].axvline(x=gd)
        ax[1, 0].set_title('Goal Difference: {:0.0f}'.format(gd))
        ax[1, 0].grid(True)
        # x, y = p_plot(self.goals_per_team[self.team_id[team_name], :])
        team_name = team.name
        p0 = self.current_points[team_name]

        i = self.team_id[team_name]

        # p0=league.current_points['Wolverhampton']
        # i = league.team_id['Wolverhampton']
        x, y = p_plot(self.points_per_team[self.team_id[team_name], :])
        ax[0, 1].bar(x, y)
        # ax[0, 1].bar(self.current_points[team_name], y.max())
        ax[0, 1].axvline(x=self.current_points[team_name])
        # xticks = ax[0, 1].get_xticks()

        P = (self.points_per_team[i, :] - p0).astype(int)
        pp = np.unique(P)
        n_to_play = len([x for x in self.matches_to_sim if team_name in [x.home_team.name, x.away_team.name]])
        pp = np.arange(n_to_play*3+1)
        probs = dict()
        for plc in places:
            probs[plc]=[]
        #prob4 = []
        #prob3 = []
        #prob2 = []
        #prob1 = []
        probp = []
        p_x = []
        for p in pp:
            ind = (P == p)
            probp.append(ind.sum() / ind.size)
            if probp[-1]>0.1/100:
                p_x.append(p+p0)
                for _plc,_prob in probs.items():
                    _prob.append(100 * np.sum(self.place_per_team[i, ind] <= _plc) / ind.sum())

                #prob4.append(100*np.sum(self.place_per_team[i, ind] <= 4) / ind.sum())
                #prob3.append(100*np.sum(self.place_per_team[i, ind] <= 3) / ind.sum())
                #prob2.append(100*np.sum(self.place_per_team[i, ind] <= 2) / ind.sum())
                #prob1.append(100 * np.sum(self.place_per_team[i, ind] <= 1) / ind.sum())

        # C = len([x for x in self.matches_to_sim if team_name in [x.home_team.name, x.away_team.name]])
        C = 1
        p_x=np.array(p_x)
        #ax[0, 1].plot(p_x / C, prob1, '.-', label='<=1')
        #ax[0, 1].plot(p_x / C, prob2, '.-', label='<=2')
        #ax[0, 1].plot(p_x / C, prob3,'.-', label='<=3')
        #ax[0, 1].plot(p_x / C, prob4,'.-', label='<=4')
        for _plc, _prob in probs.items():
            if (np.min(_prob) < 99.99) and (np.max(_prob) > 0.01):
                ax[0, 1].plot(p_x / C, _prob, '.-', label='<={:0.0f}'.format(_plc))
        ax[0, 1].grid(True)
        # ax[0, 1].plot(p_x / C, prob5,'.-', label='<=5')

        #ax[1, 1].set_title('Probabilities')
        ax[0, 1].legend()
        #ax[1, 1].set_xlim([0,n_to_play*3])
        n = 10
        a = np.min(x)
        b = np.max(x)
        if b - a > n:
            d = np.ceil((b - a) / (n - 1))
            xticks = np.arange(a, b + d, d)
        else:
            xticks = np.arange(a, b + 1)
        ax[0, 1].set_xticks(xticks)
        labels = ['+{:0.0f} = {:0.1f}/g'.format(x - self.current_points[team_name],( x - self.current_points[team_name])/n_to_play) for x in xticks]
        ax[0, 1].set_xticklabels(labels, rotation=90)
        ax[0, 1].set_title('Points: {:0.0f}'.format(self.current_points[team_name]))
        ax1=ax[1,1]
        ax2 = ax1.twinx()
        colors = ['C0','C1']
        team.plt(ax=[ax1, ax2], colors=colors)
        ax2.set_ylabel('Defense', color=colors[1])  # we already handled the x-label with ax1
        ax2.tick_params(axis='y', labelcolor=colors[1])
        ax1.set_ylabel('Offense', color=colors[0])  # we already handled the x-label with ax1
        ax1.tick_params(axis='y', labelcolor=colors[0])
        ax1.grid(color=colors[0], alpha=0.5)
        ax2.grid(color=colors[1], alpha=0.5)

        fig.set_size_inches(16*1.5, 9*1.5)
        if file_name is not None:
            folder = os.path.join(self.output_folder, team_name.replace(' ',''))
            if not os.path.isdir(folder):
                os.mkdir(folder)
            if self.as_of is None:
                file_name = self.today_str + '_' + file_name
            else:
                file_name = self.as_of.strftime('%Y-%m-%d') + '_' + file_name
            file_name = os.path.join(folder, file_name+'.png')
            fig.savefig(file_name)
            plt.close(fig=fig)


