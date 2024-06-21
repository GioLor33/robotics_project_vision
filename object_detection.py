from ultralytics import YOLO
import cv2

class Object_Detection():
    def __init__(self, model, path_to_predictions, save_image=False):
        self.model = YOLO(model)
        self.save_image = save_image

        self.detected_objects = None

        if save_image:
            assert path_to_predictions is not None
            self.path_to_predictions = path_to_predictions

    def print(self, print_to_terminal=False):
        """
        This function creates a formatteed string with all the information contained in self.detected_objects.
        
        Arguments:
            print_to_terminal:  if True, the function directly prints the formatted string to terminal
        Returns:
            text_to_print:      a formatted string with all the information provided by the model on the last image predicted
        """

        text_to_print = "\n********************\n* Objects found:"
        count = 1
        for block in self.detected_objects:
            text_to_print += f"\
                  * object {count}\n\
                  |- label: {block[6]}\n\
                  |- class: {(int(block[5]))}\n\
                  |- box: {block[:4]}\n\
                  |- confidence: {block[4]}\n\n\
                "
            count += 1
        text_to_print += "********************"

        if print_to_terminal:
            print(text_to_print)

        return text_to_print

    def predict(self, image, save_predictions_as, print_to_console=False, height=1080, width=1920, bottom_crop=0, top_crop=0, right_crop=0, left_crop=0):
        """
        This function calls the model on the image given, and perform the object detection task.

        Arguments:
            image:                  an Image object on which perform object detection
            save_predictions_as:    the (path/)name the predictions files needs to be save as
            print_to_console:       if True, prints the predicted informations on the terminal
            height:                 height of the image
            width:                  width of the image
            bottom_crop:            crop to do on the bottom part of the image, if needed
            top_crop:               crop to do on the top part of the image, if needed
            right_crop:             crop to do on the right part of the image, if needed
            left_crop:              crop to do on the left part of the image, if needed

        Returns:
            a list of objects containing 
            - [0] value of x1
            - [1] value of y1
            - [2] value of x2
            - [3] value of y2
            - [4] confidence of the model on the preediciton
            - [5] predicted class, as float
            - [6] label corresponding to the predicted class
        """

        #pre-processing of the image (cropped eliminating 370 pixels from the top in order to remote the UR5-arm)
        cropped_image = image[top_crop:height-bottom_crop, left_crop:width-right_crop]
        if self.save_image:
            cv2.imwrite(f"{self.path_to_predictions}/{save_predictions_as}.png", cropped_image)

        #prediction with the model
        results = self.model(cropped_image, imgsz=640)[0]
        
        boxes = results.boxes.xyxy.tolist()
        classes = results.boxes.cls.tolist()
        names = results.names
        confidences = results.boxes.conf.tolist()

        info_file = self.path_to_predictions + "/" + save_predictions_as + "_info.txt"
        f = open(info_file,"w")

        # Iterate through the results
        i=1
        prediction_list = []
        for box, cls, conf in zip(boxes, classes, confidences):
            x1, y1, x2, y2 = box

            y1 += 7     # the value 7 is empirically obtained
            y2 += 7     # the value 7 is empirically obtained
            
            name = names[int(cls)]

            if self.save_image:
                label = f"{name} {conf:.2f}"
                cv2.rectangle(cropped_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(cropped_image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            x1 += left_crop
            x2 += left_crop
            y1 += top_crop
            y2 += top_crop

            prediction_list.append([x1,y1,x2,y2,conf,cls,name])
            
            f.write(self.print(prediction_list))
            i+=1

        f.close()            

        # Save the resulting image with bounding boxes
        if self.save_image:
            prediction_file = f"{self.path_to_predictions}{save_predictions_as}_prediction.jpg"
            cv2.imwrite(prediction_file, cropped_image)

        if print_to_console:
            self.print(print_to_terminal=True)

        self.detected_objects = prediction_list
        return prediction_list