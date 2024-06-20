# Creation of the Object-Detection Model for the Robotics Project
This repository contains all the materials needed to train a YoloV8 model used for object detection in our [robotics project](https://github.com/nicolomarconi02/robotics-project.git) done for the course *Fundamentals of Robotics* (2023/24) held by professor Luigi Palopoli and [Michele Focchi](https://github.com/mfocchi) at University of Trento.

## Creation of the dataset
The dataset was created with [Roboflow](https://github.com/roboflow). 
We were given a set of images and labels (which can be found in `dataset/assigns`) which could not be used directly since the labels were `.json` files while Roboflow wants a `.txt` format.

To avoid this problem, we created a python script `create_annotations.py` which allowed us to recreate the `dataset/assigns` directory with the correct format required by Roboflow (more info about how this was achieved can be found in the corresponding `dataset/README.md`). The directory that we uploaded on Roboflow can be found in `dataset/readyForRoboflow`.

Thanks to Roboflow, we managed to create the dataset `dataset/dataset_RoboticsProject.yolov8` which we used to trained our object-detection model.

## Training of the model
We decided to use [YoloV8 (by Ultralytics)](https://github.com/ultralytics/ultralytics) as our object-detection model since it has better performances than [YoloV5](https://github.com/ultralytics/yolov5). 

We trained the model via a python script `model/scripts/model_train.py` (or alternatively `model/scripts/model_train.ipynb`) runned on Google Colab via GPUs. We managed to create a good model after around 1000 epochs. 

Here the diffusion matrix of the validation phase that we obtained.

![Confusion Matrix of our YoloV8 object detection model](dataset/readMe_images/confusion_matrix_normalized.png)

## Model used in the project
You can find the weights of the model we used in our robotics project in `dataset/best.py`.