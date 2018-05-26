import random
import math
import os.path

import numpy as np
import pandas as pd

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_MOVE_MINIMAP = actions.FUNCTIONS.Move_minimap.id

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index
_PLAYER_RELATIVE_MINI = features.MINIMAP_FEATURES.player_relative.index
_MINI_VISIBILITY = features.MINIMAP_FEATURES.visibility_map.index

_PLAYER_SELF = 1
_PLAYER_ENEMY = 4

_VISIBLE = 1

_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV = 45
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_BARRACKS = 21

_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [2]


ACTION_DO_NOTHING = 'donothing'
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
ACTION_BUILD_BARRACKS = 'buildbarracks'
ACTION_BUILD_MARINE = 'buildmarine'
ACTION_SCOUT = 'scout'

smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_BUILD_BARRACKS,
    ACTION_BUILD_MARINE,
]
#Split scout actions into 16 quadrants to minimize action space
for mm_x in range(0, 64):
    for mm_y in range(0, 64):
        if (mm_x + 1) % 16 == 0 and (mm_y + 1) % 16 == 0:
            smart_actions.append(ACTION_SCOUT + '_' + str(mm_x - 8) + '_' + str(mm_y - 8))


SEE_ENEMY_REWARD = 0.1
NOT_DIE_REWARD = 0.5

REWARDGL = 0

DATA_FILE = 'Scout_data'

# Stolen from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)

        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.ix[observation, :]

            # some actions have the same value
            state_action = state_action.reindex(np.random.permutation(state_action.index))

            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)

        return action

    def learn(self, s, a, r, s_):
        if s == s_:
            return
        self.check_state_exist(s_)
        self.check_state_exist(s)

        q_predict = self.q_table.ix[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.ix[s_, :].max()
        else:
            q_target = r  # next state is terminal

        # update
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))



class SmartAgent(base_agent.BaseAgent):
    ## Allows us to invert the screen and minimap and pretend all actions are from top left
    def transformDistance(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]

        return [x + x_distance, y + y_distance]

    def transformLocation(self, x, y):
        if not self.base_top_left:
            return [63 - x, 63 - y]

        return [x, y]

    def splitAction(self, action_id):
        smart_action = smart_actions[action_id]
        x = 0
        y = 0
        if '_' in smart_action:
            smart_action, x, y = smart_action.split('_')

        return (smart_action, x, y)

    def __init__(self):
        super(SmartAgent,self).__init__()
        self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))
        self.previous_action = None
        self.previous_state = None
        self.previousSupply = 0

        self.stepNum = 0
        self.CommandCenterX = None
        self.CommandCenterY = None

        if os.path.isfile(DATA_FILE + '.gz'):
            self.qlearn.q_table = pd.read_pickle(DATA_FILE + '.gz', compression='gzip')

    def step(self, obs):
        super(SmartAgent, self).step(obs)

        if obs.last():
            global REWARDGL
            print("REWARD VALUE")
            print(REWARDGL)
            self.qlearn.learn(str(self.previous_state), self.previous_action, REWARDGL, 'terminal')
            self.qlearn.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')
            self.previous_action = None
            self.previous_state = None
            self.stepNum = 0
            REWARDGL = 0
            return actions.FunctionCall(_NO_OP, [])

        unit_type = obs.observation['screen'][_UNIT_TYPE]

        if obs.first():
            player_y, player_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
            self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0
            self.previous_action = None
            self.previous_state = None
            self.previousSupply = 0

            self.stepNum = 0
            self.CommandCenterY, self.CommandCenterX = (unit_type == _TERRAN_COMMANDCENTER).nonzero()


        #############SETTING UP THE STATE#############

        depot_y, depot_x = (unit_type == _TERRAN_SUPPLY_DEPOT).nonzero()
        supply_depot_count = int(round(len(depot_y) / 69))

        barracks_y, barracks_x = (unit_type == _TERRAN_BARRACKS).nonzero()
        barracks_count = int(round(len(barracks_y) / 137))

        supply_limit = obs.observation['player'][4]
        army_supply = obs.observation['player'][8]

        if self.stepNum == 0: #if this is the first step
            self.stepNum += 1

            current_state = np.zeros(20) # Generate array of 21
            current_state[0] = supply_depot_count
            current_state[1] = barracks_count
            current_state[2] = supply_limit
            current_state[3] = army_supply

            enemy_squares = np.zeros(16)
            enemy_y, enemy_x = (obs.observation['minimap'][_PLAYER_RELATIVE_MINI] == _PLAYER_ENEMY).nonzero()
            for i in range(0,len(enemy_y)):
                y = int(math.ceil((enemy_y[i] + 1) / 16))
                x = int(math.ceil((enemy_x[i] + 1) / 16))
                enemy_squares[((y-1)*4)+(x-1)] = 1 #mark location of enemy squares


            if not self.base_top_left: #Invert the quadrants
                enemy_squares =enemy_squares[::-1]

            for i in range(0,16):
                current_state[i+4] = enemy_squares[i] #write in enemy squares location into the state

            ################################################


                #Dont learn from the first step#
            if self.previous_action is not None:
                #reward = 0
                doReward = False
                #Adjust reward based on current score
                for i in range(0,16):
                    if current_state[i+4] != self.previous_state[i+4]:
                        doReward = True
                        break

                if doReward:
                    #reward += SEE_ENEMY_REWARD
                    REWARDGL+= SEE_ENEMY_REWARD
                    #print(reward)
                self.qlearn.learn(str(self.previous_state),self.previous_action,0,str(current_state))


                    #Choose an action#
            rl_action = self.qlearn.choose_action(str(current_state))
            self.previous_state = current_state
            self.previous_action = rl_action

            smart_action,x,y = self.splitAction(self.previous_action)

            self.previousSupply = army_supply


            if smart_action == ACTION_BUILD_BARRACKS or smart_action == ACTION_BUILD_SUPPLY_DEPOT:
                unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()

                if unit_y.any():
                    i = random.randint(0, len(unit_y) - 1)
                    target = [unit_x[i], unit_y[i]]

                    return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

            elif smart_action == ACTION_BUILD_MARINE:
                if barracks_y.any():
                    i = random.randint(0, len(barracks_y) - 1)
                    target = [barracks_x[i], barracks_y[i]]

                    return actions.FunctionCall(_SELECT_POINT, [_SELECT_ALL, target])

            elif smart_action == ACTION_SCOUT:
                if _SELECT_ARMY in obs.observation['available_actions']:
                    return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])


        elif self.stepNum == 1:
            self.stepNum = 0
            smart_action,x,y = self.splitAction(self.previous_action) ##et the action

            if smart_action == ACTION_BUILD_SUPPLY_DEPOT:
                if supply_depot_count < 2 and _BUILD_SUPPLY_DEPOT in obs.observation['available_actions']:
                    if self.CommandCenterY.any():
                        if supply_depot_count == 0:
                            target = self.transformDistance(round(self.CommandCenterX.mean()), -35, round(self.CommandCenterY.mean()), 0)
                        elif supply_depot_count == 1:
                            target = self.transformDistance(round(self.CommandCenterX.mean()), -25, round(self.CommandCenterY.mean()), -25)

                        return actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [_NOT_QUEUED, target])

            elif smart_action == ACTION_BUILD_BARRACKS:
                if barracks_count < 2 and _BUILD_BARRACKS in obs.observation['available_actions']:
                    if self.CommandCenterY.any():
                        if barracks_count == 0:
                            target = self.transformDistance(round(self.CommandCenterX.mean()), 15, round(self.CommandCenterY.mean()),-9)
                        elif barracks_count == 1:
                            target = self.transformDistance(round(self.CommandCenterX.mean()), 15, round(self.CommandCenterY.mean()),12)
                        return actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])


            elif smart_action == ACTION_BUILD_MARINE:
                if _TRAIN_MARINE in obs.observation['available_actions']:
                    return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])

            elif smart_action == ACTION_SCOUT:
                do_it = True

                if len(obs.observation['single_select']) > 0 and obs.observation['single_select'][0][0] == _TERRAN_SCV:
                    do_it = False

                if len(obs.observation['multi_select']) > 0 and obs.observation['multi_select'][0][0] == _TERRAN_SCV:
                    do_it = False

                if _MOVE_MINIMAP in obs.observation["available_actions"] and do_it:
                    target = self.transformLocation((int(x)),int(y))
                    return actions.FunctionCall(_ATTACK_MINIMAP,[_NOT_QUEUED, target])


        return actions.FunctionCall(_NO_OP, [])




