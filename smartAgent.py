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
_TRAIN_REAPER = actions.FUNCTIONS.Train_Reaper_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_MOVE_MINIMAP = actions.FUNCTIONS.Move_minimap.id
_BUILD_ENGBAY = actions.FUNCTIONS.Build_EngineeringBay_screen.id
_BUILD_TURRET = actions.FUNCTIONS.Build_MissileTurret_screen.id
_HARVEST_GATHER = actions.FUNCTIONS.Harvest_Gather_screen.id
_BUILD_REFINERY = actions.FUNCTIONS.Build_Refinery_screen.id
_BUILD_TECHLAB = actions.FUNCTIONS.Build_TechLab_Barracks_quick.id

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
_TERRAN_TURRET = 23
_TERRAN_ENGBAY = 22
_NEUTRAL_MINERAL_FIELD = 341
_TERRAN_MARINE = 49
_NEUTRAL_VESPENEGEYSER = 342
_TERRAN_REFINERY = 20

_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [2]

ACTION_DO_NOTHING = 'donothing'
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
ACTION_BUILD_BARRACKS = 'buildbarracks'
ACTION_BUILD_MARINE = 'buildmarine'
ACTION_BUILD_REAPER = 'buildreaper'
ACTION_SCOUT = 'scout'
ACTION_BUILD_ENGBAY = 'buildengineeringbay'
ACTION_BUILD_TURRET = 'buildturret'
ACTION_BUILD_REFINERY = 'buildrefinery'
ACTION_BUILD_TECHLAB = 'buildtechlab'

smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_BUILD_BARRACKS,
    ACTION_BUILD_ENGBAY,
    ACTION_BUILD_TURRET,
    ACTION_BUILD_REFINERY,
    ACTION_BUILD_TECHLAB,
    ACTION_BUILD_REAPER
]
# Split scout actions into 16 quadrants to minimize action space
for mm_x in range(0, 64):
    for mm_y in range(0, 64):
        if (mm_x + 1) % 16 == 0 and (mm_y + 1) % 16 == 0:
            smart_actions.append(ACTION_SCOUT + '_' + str(mm_x - 8) + '_' + str(mm_y - 8))

SEE_ENEMY_REWARD = 0.001
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
        self.disallowed_actions = {}

    def choose_action(self, observation, excluded_actions=[]):
        self.check_state_exist(observation)

        self.disallowed_actions[observation] = excluded_actions

        state_action = self.q_table.ix[observation, :]

        for excluded_action in excluded_actions:
            del state_action[excluded_action]

        if np.random.uniform() < self.epsilon:
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

        s_rewards = self.q_table.ix[s_, :]

        if s_ in self.disallowed_actions:
            for excluded_action in self.disallowed_actions[s_]:
                del s_rewards[excluded_action]

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
                pd.Series([0] * (len(self.actions)), index=self.q_table.columns, name=state))


class SmartAgent(base_agent.BaseAgent):
    def __init__(self):
        super(SmartAgent, self).__init__()
        self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))
        self.previous_action = None
        self.previous_state = None

        self.previousSupply = 0

        self.stepNum = 0
        self.CommandCenterX = None
        self.CommandCenterY = None

        self.timeTillBase = 0
        self.baseFound = False

        if os.path.isfile(DATA_FILE + '.gz'):
            self.qlearn.q_table = pd.read_pickle(DATA_FILE + '.gz', compression='gzip')

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

    def foundBase(self,obs):
        enemy_y, enemy_x = (obs.observation['minimap'][_PLAYER_RELATIVE_MINI] == _PLAYER_ENEMY).nonzero()
        if self.base_top_left and not self.baseFound:
            found = False
            if 45 in enemy_y and 35 in enemy_x:
                found = True

            if found and not self.baseFound:
                self.baseFound = True
                print(self.timeTillBase)

        if not self.base_top_left and not self.baseFound:
            found = False
            if 25 in enemy_y and 20 in enemy_x:
                found = True

            if found and not self.baseFound:
                self.baseFound = True
                print(self.timeTillBase)


    def step(self, obs):
        super(SmartAgent, self).step(obs)

        if obs.last():
            self.obsLast()
            return actions.FunctionCall(_NO_OP, [])

        unit_type = obs.observation['screen'][_UNIT_TYPE]

        if obs.first():
            self.obsFirst(unit_type,obs)

        self.foundBase(obs)
        self.timeTillBase = self.timeTillBase + 1
        #############SETTING UP THE STATE#############
        supply_depot_count = self.supplyDepotCount(unit_type)
        cc_count = self.commandCenterCount(unit_type)
        barracks_count = self.barracksCount(unit_type)
        turrets_count = self.turretCount(unit_type)
        engbay_count = self.engbayCount(unit_type)
        refinery_count = self.refineryCount(unit_type)

        supply_used = obs.observation['player'][3]
        supply_limit = obs.observation['player'][4]
        army_supply = obs.observation['player'][5]  # check vs 8 #################
        worker_supply = obs.observation['player'][6]

        supply_free = supply_limit - supply_used


        if self.stepNum == 0:  # if this is the first step
            self.stepNum += 1
            return self.firstStep(unit_type,obs,cc_count,supply_depot_count, worker_supply, barracks_count, engbay_count,
                                               turrets_count, refinery_count, supply_free, army_supply,supply_limit)

        elif self.stepNum == 1:
            self.stepNum += 1
            return self.secondStep(unit_type,obs,cc_count,supply_depot_count, worker_supply, barracks_count, engbay_count,
                                               turrets_count, refinery_count, supply_free, army_supply,supply_limit)

        elif self.stepNum == 2:
            self.stepNum = 0
            return self.thirdStep(unit_type,obs,cc_count,supply_depot_count, worker_supply, barracks_count, engbay_count,
                                               turrets_count, refinery_count, supply_free, army_supply,supply_limit)

        return actions.FunctionCall(_NO_OP, [])


    def obsLast(self):
        global REWARDGL
       # print("REWARD VALUE")
        #print(REWARDGL)
        self.qlearn.learn(str(self.previous_state), self.previous_action, REWARDGL, 'terminal')
        self.qlearn.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')
        self.previous_action = None
        self.previous_state = None
        self.stepNum = 0
        self.kill_check = 0
        self.structure_kill = 0
        self.geyser_farm = 0
        REWARDGL = 0
        return
    def obsFirst(self,unit_type,obs):
        player_y, player_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
        self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0
        self.previous_action = None
        self.previous_state = None
        self.previousSupply = 0
        self.structure_kill = 0
        self.kill_check = 0
        self.stepNum = 0
        self.geyser_farm = 0
        self.CommandCenterY, self.CommandCenterX = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
        self.timeTillBase = 0
        self.baseFound = False
        return


    def firstStep(self,unit_type,obs,cc_count,supply_depot_count, worker_supply, barracks_count, engbay_count,
                                               turrets_count, refinery_count, supply_free, army_supply,supply_limit):
        # current state is an array holding all the state values
        current_state = self.currentState(cc_count, supply_depot_count, barracks_count, engbay_count, turrets_count,
                                          refinery_count, supply_limit, army_supply)

        # marks all the current regions with a 1 where it sees enemies
        enemy_squares = self.markEnemies(obs)

        for i in range(0, 16):
            current_state[i + 8] = enemy_squares[i]  # write in enemy squares location into the state

            # Dont learn from the first step#
        if self.previous_action is not None:
            self.learn(unit_type, obs,current_state)

        excluded_actions = self.excludeActions(supply_depot_count, worker_supply, barracks_count, engbay_count,
                                               turrets_count, refinery_count, supply_free, army_supply)
        rl_action = self.qlearn.choose_action(str(current_state), excluded_actions)
        self.previous_state = current_state
        self.previous_action = rl_action
        smart_action, x, y = self.splitAction(self.previous_action)

        self.previousSupply = army_supply

        # select SCV for building
        if smart_action == ACTION_BUILD_BARRACKS or smart_action == ACTION_BUILD_SUPPLY_DEPOT or smart_action == ACTION_BUILD_TURRET or smart_action == ACTION_BUILD_ENGBAY or smart_action == ACTION_BUILD_REFINERY:
            return self.selectSCV(unit_type)

        # selecting barracks for making marine units
        elif smart_action == ACTION_BUILD_REAPER:
            return self.selectBarracks(unit_type)

        # selecting marine units for scouting
        elif smart_action == ACTION_SCOUT:
            if _SELECT_ARMY in obs.observation['available_actions']:
                return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])
        return actions.FunctionCall(_NO_OP, [])
    def secondStep(self,unit_type,obs,cc_count,supply_depot_count, worker_supply, barracks_count, engbay_count,
                                               turrets_count, refinery_count, supply_free, army_supply,supply_limit):
        smart_action, x, y = self.splitAction(self.previous_action)  # get the action

        if smart_action == ACTION_BUILD_SUPPLY_DEPOT:
            return self.buildSupplyDepot(obs, supply_depot_count)

        elif smart_action == ACTION_BUILD_BARRACKS:
            return self.buildBarracks(obs, barracks_count)

        elif smart_action == ACTION_BUILD_ENGBAY:
            return self.buildEngbay(obs, engbay_count)

        elif smart_action == ACTION_BUILD_REFINERY:
            return self.buildRefinery(obs,refinery_count)

        elif smart_action == ACTION_BUILD_TURRET:
            return self.buildTurret(obs, turrets_count)

        elif smart_action == ACTION_BUILD_REAPER:
            return self.trainReaper(obs)

        elif smart_action == ACTION_SCOUT:
            return self.scout(obs,x,y)
        return actions.FunctionCall(_NO_OP, [])
    def thirdStep(self, unit_type, obs, cc_count, supply_depot_count, worker_supply, barracks_count, engbay_count,
                   turrets_count, refinery_count, supply_free, army_supply,supply_limit):
        smart_action, x, y = self.splitAction(self.previous_action)

        if smart_action == ACTION_BUILD_BARRACKS or smart_action == ACTION_BUILD_SUPPLY_DEPOT or smart_action == ACTION_BUILD_TURRET or smart_action == ACTION_BUILD_ENGBAY:
            if _HARVEST_GATHER in obs.observation['available_actions']:
                self.geyser_farm += 1
                if self.geyser_farm % 4 == 0:
                    unit_y, unit_x = (unit_type == _TERRAN_REFINERY).nonzero()
                    if unit_y.any():
                        i = random.randint(0, len(unit_y) - 1)

                        m_x = unit_x[i]
                        m_y = unit_y[i]

                        target = [int(m_x), int(m_y)]

                        return actions.FunctionCall(_HARVEST_GATHER, [_QUEUED, target])
                else:
                    unit_y, unit_x = (unit_type == _NEUTRAL_MINERAL_FIELD).nonzero()
                    if unit_y.any():
                        i = random.randint(0, len(unit_y) - 1)

                        m_x = unit_x[i]
                        m_y = unit_y[i]

                        target = [int(m_x), int(m_y)]

                        return actions.FunctionCall(_HARVEST_GATHER, [_QUEUED, target])
        return actions.FunctionCall(_NO_OP, [])


    def currentState(self,cc_count,supply_depot_count,barracks_count,engbay_count,
                     turrets_count,refinery_count,supply_limit,army_supply):
        current_state = np.zeros(24)  # Generate array of 22
        current_state[0] = cc_count
        current_state[1] = supply_depot_count
        current_state[2] = barracks_count
        current_state[3] = engbay_count
        current_state[4] = turrets_count
        current_state[5] = refinery_count
        current_state[6] = supply_limit
        current_state[7] = army_supply
        return current_state
    def markEnemies(self,obs):
        enemy_squares = np.zeros(16)
        enemy_y, enemy_x = (obs.observation['minimap'][_PLAYER_RELATIVE_MINI] == _PLAYER_ENEMY).nonzero()
        for i in range(0, len(enemy_y)):
            y = int(math.ceil((enemy_y[i] + 1) / 16))
            x = int(math.ceil((enemy_x[i] + 1) / 16))
            enemy_squares[((y - 1) * 4) + (x - 1)] = 1  # mark location of enemy squares
            if not self.base_top_left:  # Invert the quadrants
                enemy_squares = enemy_squares[::-1]
        return enemy_squares
    def learn(self,unit_type,obs,current_state):
        global REWARDGL
        unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
        enemy_y, enemy_x = (obs.observation['minimap'][_PLAYER_RELATIVE_MINI] == _PLAYER_ENEMY).nonzero()
        if enemy_y.any() and unit_y.mean() > 0 and unit_y.mean() < 1000:
            xdist = round((unit_x.mean() - enemy_x.mean()) ** 2)
            ydist = round((unit_y.mean() - enemy_y.mean()) ** 2)
            distance_multiplier = math.sqrt(xdist + ydist)
            # print("distance mult", distance_multiplier)
        else:
            distance_multiplier = 0

        killed_units = obs.observation["score_cumulative"][5]
        killed_structures = obs.observation["score_cumulative"][6]
        killbonus = 0
        structure_kill_bonus = 0
        if self.kill_check < killed_units:
            killbonus = 1
            self.kill_check = killed_units
        if self.structure_kill < killed_structures:
            structure_kill_bonus = 15
            self.structure_kill = killed_structures

        added_value = len(enemy_x) * SEE_ENEMY_REWARD * distance_multiplier + structure_kill_bonus + killbonus
        ## army_bonus = army_supply*0.01
        REWARDGL += added_value
        self.qlearn.learn(str(self.previous_state), self.previous_action, 0, str(current_state))
        return

        # Returns supply depot count
    def excludeActions(self,supply_depot_count,worker_supply,barracks_count,engbay_count,
                       turrets_count,refinery_count,supply_free,army_supply):
        excluded_actions = []
        if supply_depot_count == 3 or worker_supply == 0:
            excluded_actions.append(1)
            # supplydepots = True

        if supply_depot_count == 0 or barracks_count == 4 or worker_supply == 0:
            excluded_actions.append(2)
            # barracks = True

        if barracks_count == 0 or engbay_count == 1:
            excluded_actions.append(3)
            # engbay = True

        if engbay_count == 0 or turrets_count == 2:
            excluded_actions.append(4)

        if turrets_count == 0 or refinery_count == 2:
            excluded_actions.append(5)

        if supply_free == 0 or barracks_count == 0 or refinery_count == 0:
            excluded_actions.append(7)

        if army_supply == 0:
            for i in range(0, 16):
                excluded_actions.append(i + 8)

        return excluded_actions


    def selectSCV(self,unit_type):
        unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()

        if unit_y.any():
            i = random.randint(0, len(unit_y) - 1)
            target = [unit_x[i], unit_y[i]]

            return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
        return actions.FunctionCall(_NO_OP, [])
    def selectBarracks(self,unit_type):
        barracks_y, barracks_x = (unit_type == _TERRAN_BARRACKS).nonzero()
        if barracks_y.any():
            i = random.randint(0, len(barracks_y) - 1)
            target = [barracks_x[i], barracks_y[i]]
            return actions.FunctionCall(_SELECT_POINT, [_SELECT_ALL, target])
        return actions.FunctionCall(_NO_OP, [])


    def buildSupplyDepot(self,obs,supply_depot_count):
        if supply_depot_count < 3 and _BUILD_SUPPLY_DEPOT in obs.observation['available_actions']:
            if self.CommandCenterY.any():
                global REWARDGL
                if supply_depot_count == 0:
                    target = self.transformDistance(round(self.CommandCenterX.mean()), -35,
                                                    round(self.CommandCenterY.mean()), 0)
                elif supply_depot_count == 1:
                    target = self.transformDistance(round(self.CommandCenterX.mean()), -5,
                                                    round(self.CommandCenterY.mean()), -32)
                elif supply_depot_count == 2:
                    target = self.transformDistance(round(self.CommandCenterX.mean()), 13,
                                                    round(self.CommandCenterY.mean()), 0)
                    REWARDGL += 5

                return actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [_NOT_QUEUED, target])
        return actions.FunctionCall(_NO_OP, [])
    def buildBarracks(self,obs,barracks_count):
        if barracks_count < 4 and _BUILD_BARRACKS in obs.observation['available_actions']:
            if self.CommandCenterY.any():
                global REWARDGL
                if barracks_count == 0:
                    target = self.transformDistance(round(self.CommandCenterX.mean()), 32,
                                                    round(self.CommandCenterY.mean()), -20)
                    REWARDGL += 2
                elif barracks_count == 1:
                    target = self.transformDistance(round(self.CommandCenterX.mean()), 22,
                                                    round(self.CommandCenterY.mean()), -20)
                    REWARDGL += 2
                elif barracks_count == 2:
                    target = self.transformDistance(round(self.CommandCenterX.mean()), 28,
                                                    round(self.CommandCenterY.mean()), -10)
                    REWARDGL += 2
                elif barracks_count == 3:
                    target = self.transformDistance(round(self.CommandCenterX.mean()), 10,
                                                    round(self.CommandCenterY.mean()), 17)
                    REWARDGL += 4

                return actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])
        return actions.FunctionCall(_NO_OP, [])
    def buildEngbay(self,obs,engbay_count):
        if engbay_count < 1 and _BUILD_ENGBAY in obs.observation['available_actions']:
            if self.CommandCenterY.any():
                if engbay_count < 1:
                    global REWARDGL
                    target = self.transformDistance(round(self.CommandCenterX.mean()), -8,
                                                    round(self.CommandCenterY.mean()), 28)
                    REWARDGL += 5
                    return actions.FunctionCall(_BUILD_ENGBAY, [_NOT_QUEUED, target])
        return actions.FunctionCall(_NO_OP, [])
    def buildRefinery(self,obs,refinery_count):
        if refinery_count < 2 and _BUILD_REFINERY in obs.observation['available_actions']:
            if self.CommandCenterY.any():
                unit_type = obs.observation['screen'][_UNIT_TYPE]
                global REWARDGL
                if refinery_count == 0:
                    vespene_y, vespene_x = (unit_type == _NEUTRAL_VESPENEGEYSER).nonzero()
                    first_y = vespene_y[0:97]
                    first_x = vespene_x[0:97]
                    target = self.transformDistance(round(first_x.mean()), 0, round(first_y.mean()), 0)
                elif refinery_count == 1:
                    vespene_y, vespene_x = (unit_type == _NEUTRAL_VESPENEGEYSER).nonzero()
                    target = self.transformDistance(round(vespene_x.mean()), 0, round(vespene_y.mean()), 0)
                    REWARDGL += 5
                return actions.FunctionCall(_BUILD_REFINERY, [_NOT_QUEUED, target])
        return actions.FunctionCall(_NO_OP, [])
    def buildTurret(self,obs,turrets_count):
        if turrets_count < 2 and _BUILD_TURRET in obs.observation['available_actions']:
            if self.CommandCenterY.any():
                global REWARDGL
                if turrets_count == 0:
                    target = self.transformDistance(round(self.CommandCenterX.mean()), 29,
                                                    round(self.CommandCenterY.mean()), 24)
                elif turrets_count == 1:
                    target = self.transformDistance(round(self.CommandCenterX.mean()), 24,
                                                    round(self.CommandCenterY.mean()), 29)
                    REWARDGL += 5
                return actions.FunctionCall(_BUILD_TURRET, [_NOT_QUEUED, target])
        return actions.FunctionCall(_NO_OP, [])
    def trainReaper(self,obs):
        if _TRAIN_REAPER in obs.observation['available_actions']:
            global REWARDGL
            REWARDGL += 1
            return actions.FunctionCall(_TRAIN_REAPER, [_QUEUED])
        return actions.FunctionCall(_NO_OP, [])
    def scout(self,obs,x,y):
        do_it = True

        if len(obs.observation['single_select']) > 0 and obs.observation['single_select'][0][0] == _TERRAN_SCV:
            do_it = False

        if len(obs.observation['multi_select']) > 0 and obs.observation['multi_select'][0][0] == _TERRAN_SCV:
            do_it = False

        if _MOVE_MINIMAP in obs.observation["available_actions"] and do_it:
            target = self.transformLocation((int(x)), int(y))
            return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, target])
        return actions.FunctionCall(_NO_OP, [])


    def supplyDepotCount(self,unit_type):
        depot_y, depot_x = (unit_type == _TERRAN_SUPPLY_DEPOT).nonzero()
        return int(round(len(depot_y) / 69)) #69 is the size of the depot in pixels

            #returns commandCenter count
    def commandCenterCount(self,unit_type):
        cc_y, cc_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
        cc_count = 1 if cc_y.any() else 0
        return cc_count
            #Returns barracks count
    def barracksCount(self,unit_type):
        barracks_y, barracks_x = (unit_type == _TERRAN_BARRACKS).nonzero()
        return int(round(len(barracks_y) / 137))

            #returns # of turrets
    def turretCount(self,unit_type):
        turrets_y, turrets_x = (unit_type == _TERRAN_TURRET).nonzero()
        return int(round(len(turrets_y) / 52))

            #returns # of engbays
    def engbayCount(self,unit_type):
        engbay_y, engbay_x = (unit_type == _TERRAN_ENGBAY).nonzero()
        engbay_count = 1 if engbay_y.any() else 0
        return engbay_count

            #returns # of refineries
    def refineryCount(self,unit_type):
        refinery_y, refinery_x = (unit_type == _TERRAN_REFINERY).nonzero()
        return int(round(len(refinery_y) / 97))