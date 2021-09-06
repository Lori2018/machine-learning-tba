from tba import tba
import numpy as np
from match import Team, Match

pitt_key, pembroke_key, ash_key = '2018ncgre', '2018ncpem', '2018ncash'
forsyth_key, state_champs = '2018ncwin', '2018nccmp'
nc_events = ['2018ncgre', '2018ncpem', '2018ncash', '2018ncwin', '2018nccmp']
fnc_key = '2018nc'
tba.auth_key = 'vbxm8opdrSZqgnjzor6lVtLuZKTpre4oo2WR3Zw8iS3NmmI9p1G83sgC59ZmB9eF'
states = Match(state_champs)

nc_teams = [Team.keyToNum(key) for key in tba._fetch('district/%s/teams/keys' % (fnc_key))]
state_teams = [Team.keyToNum(key) for key in tba._fetch('event/%s/teams/keys' % (state_champs))]

# 32 teams, 10 features
bg_data = np.empty((len(nc_teams), 10))
index = 0
for row in bg_data:
    row = Team.compile_stats(nc_teams[index])
    index += 1

# assemble training data

# assemble training labels

# assemble validation data

# assemble validation labels