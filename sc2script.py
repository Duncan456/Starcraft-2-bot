import sc2reader
import sys
from datetime import datetime
from collections import defaultdict
from traceback import print_exc
from sc2reader.engine.plugins import APMTracker, ContextLoader, SelectionTracker
from sc2reader.events import PlayerStatsEvent, UnitBornEvent, UnitDiedEvent, UnitDoneEvent, UnitTypeChangeEvent, UpgradeCompleteEvent

def main():
    path = sys.argv[1]
    replay = sc2reader.load_replay(path, load_level=4)
    print(replay.filename)
    print(replay.release_string)
    print(replay.category)
    print(replay.end_time)
    print(replay.type)
    print(replay.game_length)
    print(replay.teams)

    #unit = sc2reader.Unit(45)

main()
