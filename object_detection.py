"""!
Contains the class Object Detection, which uses a YoloV8 fine-tuned model to detect lego bricks. 
The class provides some methods to manage the predictions as well.
"""

from ultralytics import YOLO
import cv2
import os
from itertools import combinations
import math

MAX_OVERLAP_RATE = 0.7

class Object_Detection():
    def __init__(self, model):
        self.model = YOLO(model)

    def print(self, detected_objects, print_to_terminal=False):
        """!
        This function creates a formatted string with all the information contained in detected_objects.
        
        @type: list
        @param detected_objects: a list of detected object to print

        @type: boolean
        @param print_to_terminal: if True, the function prints directly the formatted string to terminal

        @rtype: string
        @return: Returns a formatted string with all the information provided by the model on the last image predicted
        """

        text_to_print = "\n********************\n Objects found:\n\n"
        count = 1
        for block in detected_objects:
            text_to_print += f"   object {count}\n   |- label: {block[6]}\n   |- class: {(int(block[5]))}\n   |- box: {block[:4]}\n   |- confidence: {block[4]}\n\n"
            count += 1
        text_to_print += "********************"

        if print_to_terminal:
            print(text_to_print)

        return text_to_print

    def predict(self, image, path_to_save_prediction, print_to_console=False, height=1080, width=1920, bottom_crop=0, top_crop=0, right_crop=0, left_crop=0, plot_predictions=True):
        """!
        This function calls the model on the image given, and perform the object detection task.

        @type: image
        @param image: an Image object on which perform object detection

        @type: string
        @param save_predictions_as: the (path/)name the predictions files needs to be save as

        @type: boolean
        @param print_to_console: if True, prints the predicted informations on the terminal
           
        @type: float
        @param height: height of the image

        @type: float
        @param width: width of the image

        @type: int
        @param bottom_crop: pixel to crop on the bottom part of the image

        @type: int
        @param top_crop: pixel to crop on the top part of the image

        @type: int
        @param right_crop: pixel to crop on the right part of the image

        @type: int
        @param left_crop: pixel to crop on the left part of the image

        @type: boolean
        @param plot_predictions: if True, plot the predictions in the save_predictions_as directory

        @rtype: list
        @return Returns a list of objects containing (x1,x2,y2,confidence,class,label_class)
        """

        #create a directory inside path_to_save_predictions to aggregate all the prediction outputs
        os.makedirs(path_to_save_prediction)

        #pre-processing of the image (cropped eliminating 370 pixels from the top in order to remote the UR5-arm)
        cropped_image = image[top_crop:height-bottom_crop, left_crop:width-right_crop]
        cv2.imwrite(f"{path_to_save_prediction}/cropped.png", cropped_image)

        #prediction with the model
        results = self.model(cropped_image, imgsz=640)[0]
        
        boxes = results.boxes.xyxy.tolist()
        classes = results.boxes.cls.tolist()
        names = results.names
        confidences = results.boxes.conf.tolist()

        # Iterate through the results
        cropped_image_obj = cropped_image.copy()
        prediction_list = []
        for box, cls, conf in zip(boxes, classes, confidences):
            x1, y1, x2, y2 = box

            y1 += 7     # the value 7 is empirically obtained
            y2 += 7     # the value 7 is empirically obtained
            
            name = names[int(cls)]

            x1 += left_crop
            x2 += left_crop
            y1 += top_crop
            y2 += top_crop

            prediction_list.append([x1,y1,x2,y2,conf,cls,name])

        prediction_list = self.filter_predictions(prediction_list)

        info_file = path_to_save_prediction + "/info.txt"
        f = open(info_file,"w")
        f.write(self.print(prediction_list))
        f.close()

        i=1
        if plot_predictions:
            for block in prediction_list:
                x1,y1,x2,y2,conf,cls,name = block

                x1 -= left_crop
                x2 -= left_crop
                y1 -= top_crop
                y2 -= top_crop
                
                #save image with bounding-boxes
                label = f"{name} {conf:.2f}"
                cv2.rectangle(cropped_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(cropped_image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                label_obj = f"obj{i}"
                cv2.rectangle(cropped_image_obj, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(cropped_image_obj, label_obj, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                i += 1

            # Save the resulting image with bounding boxes
            prediction_file = f"{path_to_save_prediction}/prediction_name.png"
            cv2.imwrite(prediction_file, cropped_image)

            prediction_file_obj = f"{path_to_save_prediction}/prediction_obj.png"
            cv2.imwrite(prediction_file_obj, cropped_image_obj)
        
        if print_to_console:
            self.print(prediction_list, print_to_terminal=True)

        return prediction_list
    
    def filter_predictions(self, predicted_objects):
        """!
        Removes overlapping predictions that overlap for major part of their areas. The prediction kept is the one with most confidence
        
        @type: list
        @param predicted_objects: a list of predicted object

        @rtype: list
        @return: Returns the input list filtered
        """
        prediction_combinations = combinations(predicted_objects, 2)
        predictions_to_remove = []

        for (prediction_a, prediction_b) in prediction_combinations:
            x1a, y1a, x2a, y2a, confidence_a = (*[math.floor(x) for x in prediction_a[:4]], prediction_a[4])
            area_a = (x2a - x1a) * (y2a - y1a)

            x1b, y1b, x2b, y2b, confidence_b = (*[math.floor(x) for x in prediction_b[:4]], prediction_b[4])
            area_b = (x2b - x1b) * (y2b - y1b)
            
            # compute intersection area
            x1_int = max(x1a, x1b)
            x2_int = min(x2a, x2b)
            y1_int = max(y1a, y1b)
            y2_int = min(y2a, y2b)
            area_int = (x2_int - x1_int) * (y2_int - y1_int)

            overlapRate = max(area_int/area_a, area_int/area_b)

            # we have to remove one of the predictions, the one with least confidence
            if overlapRate >= MAX_OVERLAP_RATE:
                to_remove = prediction_a
                if confidence_b < confidence_a:
                    to_remove = prediction_b

                # save the prediction to remove
                predictions_to_remove.append(to_remove)

                # and delete it from the coming combinations
                prediction_combinations = [
                    p for p in prediction_combinations if p[0] != to_remove and p[1] != to_remove 
                ]

        predicted_objects = [
            p for p in predicted_objects if p not in predictions_to_remove
        ]

        return predicted_objects