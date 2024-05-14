# coding=utf8
import json


def parse_geojson():
    cat_set = set()

    f = open('data/xView_train.geojson', encoding='utf8')
    for line in f:
        res_json = json.loads(line)
        print(res_json.keys())
        print(res_json['crs'])
        print(res_json['type'])
        print(res_json['name'])
        print(res_json['features'][0])
        for feature in res_json['features']:
            properties = feature['properties']
            cat_id = properties['cat_id']
            cat_set.add(cat_id)
        print(cat_set)
    f.close()


if __name__ == '__main__':
    a = ['Apron', 'BaseballField', 'BasketballField', 'Beach', 'Bridge', 'Cemetery', 'Commercial', 'Farmland', 'Woodland', 'GolfCourse', 'Greenhouse', 'Helipad', 'LakePond', 'OilFiled', 'Orchard', 'ParkingLot', 'Park', 'Pier', 'Port', 'Quarry', 'Railway', 'Residential', 'River', 'Roundabout', 'Runway', 'Soccer', 'SolarPannel', 'SparseShrub', 'Stadium', 'StorageTank', 'TennisCourt', 'TrainStation', 'WastewaterPlant', 'WindTrubine', 'Works', 'Sea']
    a.sort()
    print(a)
