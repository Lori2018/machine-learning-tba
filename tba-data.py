from tba import tba
import numpy as np

teamkey, state_champs = 'frc1533', '2018nccmp'
tba.auth_key = 'vbxm8opdrSZqgnjzor6lVtLuZKTpre4oo2WR3Zw8iS3NmmI9p1G83sgC59ZmB9eF'
champ_teams = {} # dictionary

champs_data = tba._fetch('event/%s/teams/simple' % (state_champs))

def keyToNum(key):
    return int(key[3:])

class Match:    
    # make a new instance for different events
    def __init__(self, event_key):
        self.matches = tba._fetch('event/%s/matches' % (event_key))

    # returns the 3 teams for the alliance specified
    def get_teams(self, match_num, color):
        keys = self.matches[match_num - 1]['alliances'][color]['team_keys']
        nums = [0, 0, 0]
        index = 0
        for key in keys:
            nums[index] = keyToNum(key)
            index += 1
        return nums
    
    def get_color(self, match_num, team_num):
        blue = self.get_teams(match_num, 'blue')
        red = self.get_teams(match_num, 'red')

        if team_num in blue:
            return 'blue'
        else:
            return 'red'

    def get_robot_num(self, match_num, team_num):
        color = self.get_color(match_num, team_num)
        teams = self.get_teams(match_num, color)
        return (teams.index(team_num) + 1)

    # if the robot consistently moves off of the line in autonomous
    # maximum - 5 points
    def get_auto_run(self, match_num, team_num):
        color = self.get_color(match_num, team_num)
        return (self.matches[match_num - 1]['score_breakdown'][color]['autoRunPoints'] / 3)
    
    # assesses if team has solid programming - can score autonomously + quickly
    # + consistently
    def get_auto_scale(self, match_num, team_num):
        data = self.retrieve_data(match_num, team_num, 'autoScaleOwnershipSec')
        # auto scale points = sec * 2
        # divide by 3 to get avg per team (3 teams per alliance)
        return (data * 2 / 3)
    
    def get_auto_switch(self, match_num, team_num):
        data = self.retrieve_data(match_num, team_num, 'autoSwitchOwnershipSec')
        return (data * 2 / 3)
    
    def get_teleop_scale(self, match_num, team_num):
        data = self.retrieve_data(match_num, team_num, 'teleopScaleOwnershipSec')
        return (data / 3)
    
    def get_teleop_switch(self, match_num, team_num):
        data = self.retrieve_data(match_num, team_num, 'teleopSwitchOwnershipSec')
        return (data / 3)

    def get_endgame(self, match_num, team_num):
        data = self.retrieve_data(match_num, team_num, 'endgameRobot%s' 
            % (self.get_robot_num(match_num, team_num)))
        
        # convert to number representation
        # scale of 5 -> 5 is climbing (best) and 1 is None
        score = 1
        if data == 'Climbing':
            score = 5
        elif data == 'Parking':
            score = 2.5
        return score

    # significant amount of fouls means team does not understand the game
    def get_fouls(self, match_num, team_num):
        data = self.retrieve_data(match_num, team_num, 'foulCount')
        return (data / 3)
    
    # small number of fouls means team is active in defense most likely
    # big gap b/t alliance indicates high scoring + good defense
    # def get_defense

    # to reduce repeating data
    def retrieve_data(self, match_num, team_num, data_name):
        color = self.get_color(match_num, team_num)
        return (self.matches[match_num - 1]['score_breakdown'][color][data_name])




# make array of team numbers and names
index = 0
for data in champs_data:
    temp_dict = {}
    temp_dict["nickname"] = data["nickname"]
    champ_teams[data["team_number"]] = temp_dict

myMatch = Match(state_champs)
print(myMatch.get_teams(1, 'blue'))
print(myMatch.get_color(1, 1533))
print(myMatch.get_robot_num(1, 1533))
print(myMatch.get_teleop_switch(1, 1533))
print(myMatch.get_endgame(1, 4561))