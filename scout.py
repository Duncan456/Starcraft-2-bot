import random

import sc2
from sc2 import Race, Difficulty
from sc2.constants import *
from sc2.ids.buff_id import BuffId
from sc2.player import Bot, Computer

class ScoutBot(sc2.BotAI):




def main():
    sc2.run_game(sc2.maps.get("Abyssal Reef LE"), [
        Bot(Race.Terran, ScoutBot()),
        Computer(Race.Protoss, Difficulty.Easy)
    ], realtime=False)

if __name__ == '__main__':
    main()
