import json
import os
import re

#########################################
##### VARIABLES THAT CAN BE CHANGED #####

# Definition of the paths use to recreate the directories to plug directly on roboflow
PATH = os.getcwd()
NAME_DIRECTORY = "/assigns/"
NAME_DATASET = "/readyForRoboflow/" 
DELETE_EXISTING_DATASET = False
NAME_LABELMAP = "label.labels"

# Images variables
PATTERN = r'^view=\d+\.jpeg'    # names of the images
HEIGHT = 1024
WIDTH = 1024

LABELS_TO_CATEGORIES = {
    "X1-Y1-Z2" : 0, 
    "X1-Y2-Z2-CHAMFER" : 1, 
    "X1-Y2-Z1" : 2,
    "X1-Y2-Z2" : 3,
    "X1-Y2-Z2-TWINFILLET" : 4, 
    "X2-Y2-Z2" : 5, 
    "X2-Y2-Z2-FILLET" : 6,
    "X1-Y3-Z2" : 7,
    "X1-Y3-Z2-FILLET" : 8, 
    "X1-Y4-Z2" : 9, 
    "X1-Y4-Z1" : 10 
}

##### END: VARIABLES THAT CAN BE CHANGED #####
##############################################


def extract_data(destination_file_no_extension, data):
    # where data is a json file
    
    # we need to create a file .txt with, for each row (that indicates a different obkect in the image) we get a line such as
    # <class_id> <center_x> <center_y> <width> <height>

    # modify the name to create the file .json

    f = open(destination_file_no_extension + ".txt", "w")
    for obj in data.values():
        class_id = LABELS_TO_CATEGORIES[obj["y"]]

        ## THE CODE BELOW IS NOT WORKING BECAUSE THE BBOX GIVEN ARE WRONG !! ##
        
        #the bbox is expressed as [<minLongitude>, <min_Latitude>, <max_Longitude>, <max_Latitude>]
        #bbox = obj["bbox"]
        #height = abs(bbox[3]-bbox[1])
        #width = abs(bbox[2]-bbox[0])
        #center_x = bbox[0] + width/2
        #center_y = bbox[1] + height/2

        ## END OF CODE NOT WORKING ##

        #instead of considering the bbox given, we will analyze all the vertex of each object and we will take the coordinate of the max and the min inboth x and y axis
        x1,y1,x2,y2 = 1024,1024,0,0
        for (x,y) in obj["vertices"]:
            x1 = min(x1, x)
            y1 = min(y1, y)
            x2 = max(x2, x)
            y2 = max(y2, y)
        
        width = abs(x2-x1)
        height = abs(y2 - y1)
        center_x = x1 + width/2
        center_y = y1 + height/2

        #normalized values
        norm_width = width/WIDTH
        norm_height = height/HEIGHT
        norm_center_x = center_x/WIDTH
        norm_center_y = center_y/HEIGHT

        f.write(f"{class_id} {norm_center_x} {norm_center_y} {norm_width} {norm_height}\n")
        
    f.close()

def analyze_directory():

    for subdir, dirs, files in os.walk(PATH + NAME_DIRECTORY):

        for file in files:
            path_file = subdir + "/" + file

            remove_extension = os.path.splitext(path_file)
            destination = PATH + NAME_DATASET + f"{hash(remove_extension[0])}"

            if file.endswith('.json'):
                data = json.load(open(path_file))
                extract_data(destination, data)

            elif re.fullmatch(PATTERN, file):
                cmd = f"cp {path_file} {destination}.jpeg"
                os.system(cmd)
                
def main():
    try:
        assert os.path.isdir(PATH + NAME_DIRECTORY)

        if DELETE_EXISTING_DATASET:
            if os.path.isdir(PATH + NAME_DATASET):
                cmd = f"rm -r {PATH + NAME_DATASET}"
                os.system(cmd)
        else:
            assert not os.path.isdir(PATH + NAME_DATASET)

        #create directory for roboflow 
        os.mkdir(PATH + NAME_DATASET)

         #create the labelMap
        labelMap_path = PATH + NAME_DATASET + NAME_LABELMAP
        f = open(labelMap_path,"w")
        for class_element in LABELS_TO_CATEGORIES:
            f.write(class_element+"\n")
        f.close()

        analyze_directory()

    except:
        print(f"This ERROR could be due to the fact that: \n-[{PATH + NAME_DIRECTORY}] is not the right path for the directory that contains the image to label \n-[{PATH + NAME_DATASET}] already exists as a directory")    

if __name__ == "__main__":

    main()