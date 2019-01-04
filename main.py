import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from football_sim.football_sim import *
import os

calibrator = Calibrator('calibration.pkl', redo=True)
calibrator.create_all_teams()
calibrator.process_data()
teams = calibrator.teams
