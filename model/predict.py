from ultralytics import YOLO
import os
import cv2

# Load a model
model = YOLO("best.pt")  # pretrained custom model

# images sizes
HEIGHT : int = 1080   #pixels
WIDTH : int = 1920    #pixels
BOTTOM_CROP : int = 0
TOP_CROP : int = 370  # pixels
RIGHT_CROP : int = 0
LEFT_CROP : int = 0

# Run batched inference on a list of images
PATH_TO_IMAGES = "/home/giolor/trento_lab_home/ros_ws/src/model-vision/model/images/"
PATH_TO_PREDICTIONS = "/home/giolor/trento_lab_home/ros_ws/src/model-vision/model/predictions/"

CONFIDENCE_THRESHOLD = 0.5
INTERSECTION_OVER_UNION_THRESHOLD = 0.5

for sub,dir,files in os.walk(PATH_TO_IMAGES):
    for file in files:

        #pre-processing of the image (cropped eliminating 370 pixels from the top in order to remote the UR5-arm)
        img = cv2.imread(f"{PATH_TO_IMAGES}{file}")
        cropped_image = img[TOP_CROP:HEIGHT-BOTTOM_CROP, LEFT_CROP:WIDTH-RIGHT_CROP]
        cv2.imwrite(f"{PATH_TO_PREDICTIONS}{file}", cropped_image)

        #prediction with the model
        results = model(cropped_image, imgsz=640, conf=CONFIDENCE_THRESHOLD, iou=INTERSECTION_OVER_UNION_THRESHOLD)
        #results = model(f"{PATH_TO_IMAGES}{file}")  # return a list of Results objects
        boxes = results[0].boxes.xyxy.tolist()
        classes = results[0].boxes.cls.tolist()
        names = results[0].names
        confidences = results[0].boxes.conf.tolist()

        name_file = os.path.splitext(file)
        box_file = name_file[0] + "_info.txt"
        f = open(PATH_TO_PREDICTIONS + box_file,"w")
        
        # Iterate through the results
        i=1
        prediction_list = []
        for box, cls, conf in zip(boxes, classes, confidences):
            x1, y1, x2, y2 = box
            
            name = names[int(cls)]

            label = f"{name} {conf:.2f}"
            cv2.rectangle(cropped_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(cropped_image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            x1 += LEFT_CROP
            x2 += LEFT_CROP
            y1 += TOP_CROP
            y2 += TOP_CROP

            prediction_list.append([x1,y1,x2,y2,confidences,cls,name])
            
            f.write(f"OBJ-{i}\nbox: [{x1},{y1},{x2},{y2}]\nconf: {conf}\ndetected_class: {cls}\nname: {name}\n\n*************************************************\n\n")
            i+=1

        f.close()            

        # Save the resulting image with bounding boxes
        prediction_file = f"{PATH_TO_PREDICTIONS}{name_file[0]}_prediction.png"
        cv2.imwrite(prediction_file, cropped_image)
