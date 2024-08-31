"""!
Contains the class Object Detection, which uses a YoloV8 fine-tuned model to detect lego bricks. 
The class provides some methods to manage the predictions as well.
"""

from ultralytics import YOLO
import cv2
import os
import numpy as np

class Object_Detection():
    def __init__(self, model):
        self.model = YOLO(model)

    def print(self, detected_objects, print_to_terminal=False):
        """!
        This function creates a formatted string with all the information contained in detected_objects.
        
        @param detected_objects: a list of detected object to print
        @param print_to_terminal: if True, the function prints directly the formatted string to terminal

        @return Returns a formatted string with all the information provided by the model on the last image predicted
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

        @param image: an Image object on which perform object detection
        @param save_predictions_as: the (path/)name the predictions files needs to be save as
        @param print_to_console: if True, prints the predicted informations on the terminal
        @param height: height of the image
        @param width: width of the image
        @param bottom_crop: pixel to crop on the bottom part of the image
        @param top_crop: pixel to crop on the top part of the image
        @param right_crop: pixel to crop on the right part of the image
        @param left_crop: pixel to crop on the left part of the image
        @param plot_predictions: if True, plot the predictions in the save_predictions_as directory

        @return Returns a list of objects containing (x1,x2,y2,confidence,class,label_class)
        """

        #create a directory inside path_to_save_predictions to aggregate all the prediction outputs
        os.makedirs(path_to_save_prediction)

        image = self.filter_image(image)
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
    
    def filter_image(self, image_cv2):
        """! 
        Removes all unwanted factors from the image, such as the background, the table shape and the shadows.

        @param imgave_cv2: the image to filter
        @return Returns a filtered image
        """

        # we have to cut above the line passing through the points
        # (1200,410) and (1540,900)
        # to exclude the blocks already positioned
        for (py, row) in enumerate(image_cv2):
            for (px,pixel) in enumerate(row):
                if ( (px - 1200) / 340 - (py - 410) / 490 ) > 0:
                    image_cv2[py][px] = (255, 255, 255)

        hsv = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, (0, 0, 100), (255, 5, 255))

        # Build mask of non black pixels.
        nzmask = cv2.inRange(hsv, (0, 0, 5), (255, 255, 255))

        # Erode the mask - all pixels around a black pixels should not be masked.
        nzmask = cv2.erode(nzmask, np.ones((3,3)))

        mask = mask & nzmask

        image_cv2[np.where(mask)] = 255

        return image_cv2