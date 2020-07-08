import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd

import datetime

headers = {'User-Agent':
               'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'}


def map_to_num(x):
    if '%' in x:
        return int(x.replace('%', '')) / 100
    else:
        return int(x)


class Team:
    def __init__(self, name):
        self.name = name
        self.url = None

    def set_url(self, url):
        self.url = url

    def __repr__(self):
        return self.name


class Fixture:
    def __init__(self, home_team, away_team):
        self.home = home_team
        self.away = away_team
        self.name = '{:s} v {:s}'.format(self.home.name, self.away.name)
        self.date = None
        self.home_goals = None
        self.away_goals = None
        self.url = None
        self.stats = None

    def get_stats(self, force_refresh=False):
        if (self.url is not None) and ((self.stats is None) or force_refresh):
            _tree = requests.get(self.url, headers=headers)
            _soup = BeautifulSoup(_tree.content, 'html.parser')
            stat_names = ['Possession', 'Shots', 'Cards', 'Corners', 'Fouls', 'Offsides']
            temp = _soup.find_all('div', {'class': lambda x: x and x.startswith('w100 cf')})
            self.stats = dict()
            for x in temp:
                z = [y.text for y in x.find_all('div')]
                if (len(z) == 8) and (z[0] in stat_names):
                    stat_name = z[0]
                    stat_home = map_to_num(z[1])
                    stat_away = map_to_num(z[7])
                    self.stats[stat_name] = (stat_home, stat_away)
                    # print(stat_name,stat_home,stat_away)

        return self.stats

    def set_date(self, date):
        self.date = date

    def set_url(self, url):
        self.url = url

    def get_des(self):

        if (self.date is None) and (self.home_goals is None):
            return self.name
        elif (self.date is None) and (self.home_goals is not None):
            return '{:s} ({:d}) v {:s} ({:d})'.format(self.home.name, self.home_goals, self.away.name, self.away_goals)
        elif (self.date is not None) and (self.home_goals is None):
            return '{:s} - {:s} v {:s} )'.format(str(self.date), self.home.name, self.away.name)
        else:
            return '{:s} - {:s} ({:d}) v {:s} ({:d})'.format(str(self.date), self.home.name, self.home_goals, self.away.name, self.away_goals)

    def set_score(self, score):
        self.home_goals = score[0]
        self.away_goals = score[1]

    def __repr__(self):
        return self.get_des()

    def __str__(self):
        return self.get_des()


class TeamSet:
    def __init__(self):
        self.teams = {}

    def add_team(self, team):
        self.teams[team.name] = team


def parse_league(url):
    pageTree = requests.get(url, headers=headers)
    pageSoup = BeautifulSoup(pageTree.content, 'html.parser')

    temp = pageSoup.find_all('tr', {'class': 'match complete'})
    team_set = TeamSet()
    fixture_dict = dict()
    for x in temp:
        home_blob = x.find('td', {'class': 'team-home'})
        home_name = home_blob.find('span').text
        home_url = 'https://footystats.org' + home_blob.find('a').attrs['href']

        if home_name in team_set.teams:
            home = team_set.teams[home_name]
        else:
            home = Team(home_name)
            home.set_url(home_url)
            team_set.add_team(home)

        away_blob = x.find('td', {'class': 'team-away'})
        away_name = away_blob.text
        away_url = 'https://footystats.org' + away_blob.find('a').attrs['href']
        if away_name in team_set.teams:
            away = team_set.teams[away_name]
        else:
            away = Team(away_name)
            away.set_url(away_url)
            team_set.add_team(away)

        status = x.find('td', {'class': 'status'})
        score = [int(x) for x in status.find('span').text.split('-')]
        link = 'https://footystats.org' + status.find('a').attrs['href']
        date = datetime.date(1970, 1, 1) + datetime.timedelta(seconds=int(x.find('td', {'class': 'date'}).attrs['data-time']))
        fixture = Fixture(home, away)
        fixture.set_score(score)
        fixture.set_url(link)
        fixture.set_date(date)
        fixture_dict[fixture.name] = fixture
        print(fixture.get_des())
    date = datetime.date.today() + datetime.timedelta(days=10)
    for h in team_set.teams.values():
        for a in team_set.teams.values():
            if h.name!=a.name:
                f = Fixture(h,a)
                f.set_date(date)
                f.set_score([-100,-100])
                if f.name not in fixture_dict:
                    fixture_dict[f.name] = f

    trash=1

    return team_set, fixture_dict

def get_data_and_merge():
    page = 'https://footystats.org/england/fa-womens-super-league/fixtures'
    # page = 'https://www.bbc.com/sport/football/51519244'
    team_set, fixture_dict = parse_league(page)
    df = pd.read_csv('https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv')
    fixtures = list(fixture_dict.values())
    date = []
    league_id = []
    league = []
    team1 = []
    team2 = []
    score1 = []
    score2 = []
    data_dict = {x: '' for x in df.columns}
    # data_dict=dict()
    for f in fixtures:
        date.append(str(f.date))
        league_id.append(df['league_id'].max() + 1)
        league.append('FA Women Super League')
        team1.append(f.home.name)
        team2.append(f.away.name)
        score1.append(f.home_goals)
        score2.append(f.away_goals)

    # str(fixture_dict['Everton Women v Arsenal Women'].date)
    data_dict['date'] = date
    data_dict['league_id'] = league_id
    data_dict['league'] = league
    data_dict['team1'] = team1
    data_dict['team2'] = team2
    data_dict['score1'] = score1
    data_dict['score2'] = score2
    pd.concat([df, pd.DataFrame(data_dict)], axis=0).to_csv('spi_matches_plus.csv', index=False)