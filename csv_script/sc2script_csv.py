import argparse
import csv
import sys
from os.path import isfile, join
import sc2reader
from sc2reader.engine.plugins import SelectionTracker, APMTracker
def csv_out(results):
    for i in range(0,len(results)):

        print('unit_type: ' + results[i]['unit_type:'] + ", "
              'unit_id: ' + results[i]['unit_id:'] + ", "
              'purchase_game_second: ' + results[i]['purchase_game_second:'] + ", "
              'purchase_x_coordinate: ' + results[i]['purchase_x_coordinate:'] + ", "
              'purchase_y_coordinate: ' + results[i]['purchase_y_coordinate:'] + ", "
              'death_game_second: ' + results[i]['death_game_second:'] + ", "
              'death_x_coordinate: ' + results[i]['death_x_coordinate:'] + ", "
              'death_y_coordinate: ' + results[i]['death_y_coordinate:'] + ", "
              'killing_unit_id: ' + results[i]['killing_unit_id:'] + ", "
              'life_span: ' + results[i]['life_span:']
        )




def getData(created, deaths):
    results = []
    for i in range(0,len(created)):
        for j in range(0, len(deaths)):
            if created[i]['unit_id:'] == deaths[j]['unit_id:']:
                survival = deaths[j]['game_second:'] - created[i]['game_second:']

                game_results = {
                    'life_span:': str(survival),

                    'killing_unit_id:': str(deaths[j]['killing_unit_id:']),
                    'death_x_coordinate:': str(deaths[j]['x_coordinate:']),
                    'death_y_coordinate:': str(deaths[j]['y_coordinate:']),
                    'death_game_second:': str(deaths[j]['game_second:']),



                    'purchase_x_coordinate:': str(created[i]['x_coordinate:']),
                    'purchase_y_coordinate:': str(created[i]['y_coordinate:']),
                    'purchase_game_second:': str(created[i]['game_second:']),
                    'unit_id:': str(created[i]['unit_id:']),
                    'unit_type:': str(created[i]['unit_type:'])
                }
                # print(game_results)
                results.append(game_results)
    #print(results)
    csv_out(results)


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
                    'game_second:': event.second,
                    #'player': event.unit_controller.toon_id,
                    'unit_id:': event.unit_id
                }

                created.append(created_data)

        if type(event) is sc2reader.events.tracker.UnitDiedEvent:
            death_data = {
                'killing_unit_id:': event.killing_unit_id,
                'x_coordinate:': event.location[0],
                'y_coordinate:': event.location[1],
                'game_second:': event.second,
                #'unit_type:': event.unit_type_name,                    'game_second': event.second,
                #'player': event.unit_controller.toon_id,
                'unit_id:': event.unit_id
            }

            deaths.append(death_data)
    getData(created, deaths)


def main():
    args = sc2_parser()
    if(".sc2replay" in args.replay_in.lower()):

        extract(args.replay_in, args.out)

if __name__ == '__main__':
    main()
