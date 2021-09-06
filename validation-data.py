from tba import tba
import numpy as np
from match import Team, MatchQM

training_labels = np.loadtxt('training_labels.txt')
training_data = np.loadtxt('training_data.txt', ndmin = 3)

print(training_data)