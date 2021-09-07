import numpy as np
from match import Team, MatchQM
from tba import tba

data_file = open("training_data.txt", 'r')
raw = data_file.read()
data = raw.split('match')

# number of matches (407), teams per match, 10 features of each team
training_data = np.empty((len(data) - 1, 6, 10))

mtch_cnt = 0
for match in data:
    teams = match.split('\n')
    teams.remove('')
    
    tm_cnt = 0
    for team in teams:
        data = team.split(' ')
        data.remove('') if '' in data else None
        if len(data) > 1:
            features = [float(point) for point in data if point != '']
        if mtch_cnt < 407:
            training_data[mtch_cnt][tm_cnt] = features  
        tm_cnt = tm_cnt + 1 if tm_cnt < 5 else 0
    mtch_cnt += 1

# print(training_data)
print(training_data[406])

data_file.close()