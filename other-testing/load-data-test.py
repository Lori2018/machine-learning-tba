import numpy as np
from match import Team, Match
from tba import tba

nc_events = ['2018ncgre', '2018ncpem', '2018ncash', '2018ncwin', '2018nccmp']
fnc_key = '2018nc'
state_champs = '2018nccmp'

nc_teams = [Team.keyToNum(key) for key in tba._fetch('district/%s/teams/keys' % (fnc_key))]
state_teams = [Team.keyToNum(key) for key in tba._fetch('event/%s/teams/keys' % (state_champs))]

train_data_file = open("validation_data.txt", "w+")
train_labels_file = open("validation_labels.txt", "w+")

# 32 teams, 10 features
# dictionary for easy access
bg_data = {}
index = 0
for count in range(len(nc_teams)):
    bg_data[nc_teams[count]] = Team.compile_stats(nc_teams[count])
print("Finished creating NC team dictionary")

for event in nc_events:
    myMatch = Match(event)
    all_matches = tba._fetch('event/%s/matches' % (event))
    # 6 teams, 10 features
    mtch_dta = np.empty((6, 10))
    for match in all_matches:
        match_num = match['match_number']
        print(match_num)
        red_teams, blue_teams = myMatch.get_teams(match_num, 'red'), myMatch.get_teams(match_num, 'blue')
        for team in red_teams:
            if team in bg_data:
                for point in bg_data[team]:
                    train_data_file.write(str(point) + ' ')
                train_data_file.write('\n')
                print(bg_data[team])
            else:
                for point in range(10):
                    train_data_file.write('0')
                train_data_file.write('\n')
                print("----- Not an NC Team -----")
        for team in blue_teams:
            if team in bg_data:
                for point in bg_data[team]:
                    train_data_file.write(str(point) + ' ')
                train_data_file.write('\n')
                print(bg_data[team])
            else:
                for point in range(10):
                    train_data_file.write('0')
                train_data_file.write('\n')
                print("----- Not an NC Team -----")
        train_data_file.write('match\n')
        print("Finished a match")

        # training labels
        red_score = myMatch.red_total(match_num)
        blue_score = myMatch.blue_total(match_num)
        # 0 - red, 1 - blue
        score = blue_score / (blue_score + red_score)
        train_labels_file.write(str(score))
        train_labels_file.write("\n")
    print("------- FINISHED AN EVENT -------")
    
train_data_file.close()
train_labels_file.close()