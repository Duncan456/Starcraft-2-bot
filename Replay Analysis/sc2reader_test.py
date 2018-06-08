import argparse
import csv
import sys
from os.path import isfile, join
import sc2reader
from sc2reader.engine.plugins import SelectionTracker, APMTracker
sc2reader.engine.register_plugin(APMTracker())
sc2reader.engine.register_plugin(SelectionTracker())
from sc2reader.factories import SC2Factory

import sc2reader



def main():
    replay = sc2reader.load_replay('not_working.SC2Replay', load_level=4)
    #replay.load_map()

    print (replay)



if __name__ == '__main__':
    main()
