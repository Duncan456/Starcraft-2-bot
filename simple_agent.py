from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features
import time
import random

# Functions
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_BUILD_SUPPLYDEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_NOOP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_TRAIN_SCV = actions.FUNCTIONS.Train_SCV_quick.id
_RALLY_UNITS_MINIMAP = actions.FUNCTIONS.Rally_Units_minimap.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

# Features
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

# Unit IDs
_TERRAN_BARRACKS = 21
_TERRAN_COMMANDCENTER = 18
_TERRAN_SUPPLYDEPOT = 19
_TERRAN_SCV = 45

# Parameters
_PLAYER_SELF = 1
_SUPPLY_USED = 3
_SUPPLY_MAX = 4
_NOT_QUEUED = [0]
_QUEUED = [1]

def selectScv(self,obs):
    unit_type = obs.observation["screen"][_UNIT_TYPE] #get all units on screen
    unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero() #select terran_scv on screen
    target = [unit_x[0], unit_y[0]]
    self.scv_selected = True
    self.barracks_selected = False
    self.commandCenter_selcted = False
    self.army_selected = False
    return actions.FunctionCall(_SELECT_POINT,[_NOT_QUEUED, target])

def buildSupplyDepot(self,obs):
    unit_type = obs.observation["screen"][_UNIT_TYPE]
    unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
    target = self.transformLocation(int(unit_x.mean()),0,int(unit_y.mean()),20) #Changes the target based on where your base is
    self.supply_depot_built = True
    return actions.FunctionCall(_BUILD_SUPPLYDEPOT, [_NOT_QUEUED,target]) #build the depot at target
    
def buildBarracks(self,obs):
    unit_type = obs.observation["screen"][_UNIT_TYPE] # get unit types on screen
    unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero() # get location of commandCenter
    target = self.transformLocation(int(unit_x.mean()), 20, int(unit_y.mean()),0) #target 20 in x direction of base
    self.barracks_built = True
    return actions.FunctionCall(_BUILD_BARRACKS,[_NOT_QUEUED, target])

def selectCommandCenter(self,obs):
    unit_type = obs.observation["screen"][_UNIT_TYPE]
    unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
    target = [int(unit_x.mean()),int(unit_y.mean())]
    self.scv_selected = False
    self.barracks_selected = False
    self.commandCenter_selected = True
    return actions.FunctionCall(_SELECT_POINT,[_NOT_QUEUED,target])
    


    
    
    
class SimpleAgent(base_agent.BaseAgent):
    base_top_left = None
    supply_depot_built = False
    barracks_built = False
    
    barracks_rallied = False
    army_rallied = False

    army_selected = False
    barracks_selected = False
    scv_selected = False
    commandCenter_selected = False

        

    def step(self, obs):
        super(SimpleAgent, self).step(obs)

       # time.sleep(0.5)

        #figure out where the base is
        if self.base_top_left is None:
            player_y, player_x = (obs.observation["minimap"][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
            self.base_top_left = player_y.mean() <= 31
    
        if not self.supply_depot_built: #build supply depot if one is not built
            if not self.scv_selected: #select scv if one is not selected
                return selectScv(self,obs) #actually select unit

            elif _BUILD_SUPPLYDEPOT in obs.observation["available_actions"]: #if scv is selected
                return buildSupplyDepot(self,obs) #build supply depot
            
            return actions.FunctionCall(_NOOP,[])
        
        elif not self.barracks_built: #build a barracks, scv is already selected
            
            if _BUILD_BARRACKS in obs.observation["available_actions"]: #if building a barracks is an available action 
                return buildBarracks(self,obs)  #build barracks
            
        elif not self.barracks_rallied: #change rally point of barracks
            if not self.barracks_selected: 
                unit_type = obs.observation["screen"][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_BARRACKS).nonzero()

                if unit_y.any():
                    target = [int(unit_x.mean()), int(unit_y.mean())]
                    self.commandCenter_selected = False
                    self.barracks_selected = True
                    self.scv_selected = False
                    return actions.FunctionCall(_SELECT_POINT,[_NOT_QUEUED, target])
            else:
                self.barracks_rallied = True
                if self.base_top_left:
                    return actions.FunctionCall(_RALLY_UNITS_MINIMAP, [_NOT_QUEUED, [29, 21]])

                return actions.FunctionCall(_RALLY_UNITS_MINIMAP, [_NOT_QUEUED, [29, 46]])


        elif obs.observation["player"][_SUPPLY_USED] < obs.observation["player"][_SUPPLY_MAX]:
            if not self.commandCenter_selected:
                print("About to select command center")
                return selectCommandCenter(self,obs)
            elif _TRAIN_SCV in obs.observation["available_actions"]:
                print("About to build scv")
                return actions.FunctionCall(_TRAIN_SCV, [_QUEUED])

        elif not self.army_rallied:
            if not self.army_selected:
                if _SELECT_ARMY in obs.observation["available_actions"]:
                    self.army_selected = True
                    self.barracks_selected = False
                    self.commandCenter_selected = False
                    self.scv_selected = False

                    return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])
            elif _ATTACK_MINIMAP in obs.observation["available_actions"]:
                self.army_rallied = True
                self.army_selected = False

                if self.base_top_left:
                    return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [39, 45]])

                return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [21, 24]])


        return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])
    def transformLocation(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return[x-x_distance, y-y_distance]
        return [x+x_distance,y+y_distance]
