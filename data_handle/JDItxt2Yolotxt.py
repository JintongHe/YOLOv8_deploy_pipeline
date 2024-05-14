import os
from tqdm import tqdm
import cv2


def JDItxt2Yolotxt():
    imgtxtfiles = r"D:\xview\dota-all\dota-xview\xview_split\data_with_YOLO_format_instance_segmentation"

    imgtxtList = os.listdir(imgtxtfiles)
    imgFormat = ".png"
    imgList = [x for x in imgtxtList if x.endswith(imgFormat)]

    classLabel = ['Building', 'Small-Car', 'Truck', 'Cargo-Truck', 'Damaged-Building', 'Trailer', 'Truck-w/Flatbed', 'Bus',
                  'Passenger-Vehicle', 'Construction-Site', 'Utility-Truck', 'Maritime-Vessel', 'Motorboat', 'Sailboat',
                  'Scraper/Tractor', 'Excavator', 'Dump-Truck', 'Shipping-Container', 'Front-loader/Bulldozer',
                  'Mobile-Crane', 'Crane-Truck', 'Truck-w/Box', 'Truck-Tractor', 'Vehicle-Lot', 'Pickup-Truck',
                  'Cement-Mixer', 'Engineering-Vehicle', 'Storage-Tank', 'Ground-Grader', 'Hut/Tent', 'Facility',
                  'Fixed-wing-Aircraft', 'Reach-Stacker', 'Tower', 'Passenger-Car', 'Cargo-Car', 'Shed', 'Cargo-Plane',
                  'Shipping-container-lot', 'Locomotive', 'Pylon', 'Tank-car', 'Flat-Car', 'other', 'Railway-Vehicle',
                  'Fishing-Vessel', 'Barge', 'Tugboat', 'Haul-Truck', 'Helipad', 'Tower-crane', 'Container-Crane',
                  'Small-Aircraft', 'Helicopter', 'Oil-Tanker', 'Ferry', 'Truck-w/Liquid', 'Aircraft-Hangar', 'Yacht',
                  'Container-Ship', 'Straddle-Carrier']
    for item in tqdm(imgList):
        img = cv2.imread(os.path.join(imgtxtfiles, item))
        w = img.shape[1]
        h = img.shape[0]

        tmp = ""
        labels = ""
        with open(os.path.join(imgtxtfiles, item.replace(imgFormat, ".txt")), "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                if len(line) >= 10:
                    continue
                else:
                    label = classLabel.index(line[-1])
                    del line[-1]
                    line.insert(0, label)
                    line[3] = float(line[3]) - float(line[1])
                    line[5] = float(line[5]) - float(line[1])
                    line[6] = float(line[6]) - float(line[2])
                    line[8] = float(line[8]) - float(line[2])
                    for i in range(len(line)):
                        if i % 2 == 1:
                            line[i] = float(line[i]) / w
                            line[i + 1] = float(line[i + 1]) / h
                    line = [str(x) for x in line]
                    tmp += (" ".join(line) + "\n")

        os.remove(os.path.join(imgtxtfiles, item.replace(imgFormat, ".txt")))
        with open(os.path.join(imgtxtfiles, item.replace(imgFormat, ".txt")), "w") as t:
            t.writelines(tmp, )


if __name__ == '__main__':
    pass
