import argparse
import csv
import sys
from os.path import isfile, join
import sc2reader
from sc2reader.engine.plugins import SelectionTracker, APMTracker

def sc2_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('replay_in', metavar='PATH', type=str, help="replay file")
    parser.add_argument('out', metavar='PATH', type=str, help="csv file")
    return parser.parse_args()

def extract(in_path, out_path):
    replay = sc2reader.load_replay(in_path, load_level = 4)
    created = []
    deaths = []


    for event in replay.events:
        # Unit created
        if type(event) is sc2reader.events.tracker.UnitBornEvent:
            if event.unit_type_name in ['Marine']:
                created_data = {
                    'x_coordinate:': event.location[0],
                    'y_coordinate:': event.location[1],
                    'unit_type:': event.unit_type_name,
                    'game_second': event.second,
                    #'player': event.unit_controller.toon_id,
                    'unit_id': event.unit_id
                }

                created.append(created_data)

    print(created)





def main():
    args = sc2_parser()
    if(".sc2replay" in args.replay_in.lower()):

        extract(args.replay_in, args.out)

if __name__ == '__main__':
    main()
