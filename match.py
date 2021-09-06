from tba import tba
import numpy as np

nc_events = ['2018ncgre', '2018ncpem', '2018ncash', '2018ncwin', '2018nccmp']

class MatchQM:    
    # make a new instance for different events
    def __init__(self, event_key):
        self.event_key = event_key
        self.matches = tba._fetch('event/%s/matches' % (event_key))
        updated_matches = []
        other_matches = []
        for match in self.matches:
            if match['comp_level'] == 'qm':
                updated_matches.append(match)
            else: 
                other_matches.append(match)

        self.matches = updated_matches
        self.other_matches = other_matches

    def get_match_data(self, match_num):
        for match in self.matches:
            if match['match_number'] == match_num:
                return match

    # returns the 3 teams for the alliance specified
    def get_teams(self, match_num, color):
        match = self.get_match_data(match_num)
        keys = match['alliances'][color]['team_keys']
        nums = [0, 0, 0]
        index = 0
        for key in keys:
            nums[index] = Team.keyToNum(key)
            index += 1
        return nums
    
    def get_color(self, match_num, team_num):
        blue = self.get_teams(match_num, 'blue')
        red = self.get_teams(match_num, 'red')

        return 'blue' if team_num in blue else 'red'

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
    def get_strategy(self, match_num, team_num):
        fouls = self.get_fouls(match_num, team_num)
        gap = 0.125 * (self.red_total(match_num) - self.blue_total(match_num))
        gap *= -1 if self.get_color(match_num, team_num) == 'blue' else 1
        teleop_switch = (self.get_teleop_switch(match_num, team_num)) / 12
        return (fouls + gap + teleop_switch)

    def get_defense(self, match_num, team_num):
        data = self.retrieve_data(match_num, team_num, 'teleopScaleBoostSec') + self.retrieve_data(match_num, team_num, 'teleopScaleForceSec')
        return data

    # to reduce repeating data
    def retrieve_data(self, match_num, team_num, data_name):
        color = self.get_color(match_num, team_num)
        return (self.matches[match_num - 1]['score_breakdown'][color][data_name])

    def red_total(self, match_num):
        return self.matches[match_num - 1]['score_breakdown']['red']['totalPoints']

    def blue_total(self, match_num):
        return self.matches[match_num - 1]['score_breakdown']['blue']['totalPoints']

class Team:
    # static function
    def keyToNum(key):
        return int(key[3:])

    def numToKey(num):
        return 'frc' + str(num)

    def get_events(team_num):
        all_events = tba._fetch('team/%s/events/2018' % (Team.numToKey(team_num)))
        event_keys = [event["key"] for event in all_events]
        nc_lst = [event for event in event_keys if event in nc_events]
        return nc_lst

    def get_matches(team_num, event_key):
        matches = tba._fetch('team/%s/event/%s/matches' % (Team.numToKey(team_num), event_key))
        mtch_lst = []
        for match in matches:
            mtch_lst.append(match["match_number"]) if (match["comp_level"] == "qm") else None
        mtch_lst.sort()
        return mtch_lst

    def compile_stats(team_num):
        nc_events = Team.get_events(team_num)
        count = 0
        for event in nc_events:
            matches, stats = Team.get_matches(team_num, event), np.empty(10)
            myMatch = MatchQM(event)
            for match in matches:
                stats[0] = team_num
                stats[1] = Team.updateVal(count, stats[1], myMatch.get_auto_run(match, team_num))
                stats[2] = Team.updateVal(count, stats[2], myMatch.get_auto_scale(match, team_num))
                stats[3] = Team.updateVal(count, stats[3], myMatch.get_auto_switch(match, team_num))
                stats[4] = Team.updateVal(count, stats[4], myMatch.get_teleop_scale(match, team_num))
                stats[5] = Team.updateVal(count, stats[5], myMatch.get_teleop_switch(match, team_num))
                stats[6] = Team.updateVal(count, stats[6], myMatch.get_endgame(match, team_num))
                stats[7] = Team.updateVal(count, stats[7], myMatch.get_fouls(match, team_num))
                stats[8] = Team.updateVal(count, stats[8], myMatch.get_defense(match, team_num))
                stats[9] = Team.updateVal(count, stats[9], myMatch.get_strategy(match, team_num))
                count += 1
        return stats

    def updateVal(count, last_val, new_val):
        mult = count * last_val
        return (mult + new_val) / (count + 1)