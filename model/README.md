# Training of the model
We chose to use a [YoloV8 (nano) model](https://github.com/ultralytics/ultralytics), provided by Ultralytics, trained with a [custom dataset](https://universe.roboflow.com/giolor/robotics_project) on Google Colab. The script can be found in `scripts/model_train.ipynb`. 

## Structure of the model_train.ipynb file
Firstly, since Google Colab provides only a limited amount of GPU every day, we mounted Google Drive in our script in order to have a directory in it where to save, every *n* epochs of training, the best weights of the model obtained and the corresponding validation. The directory we worked in can be found [here](https://drive.google.com/drive/folders/1fOuXB3zEGcUwMVJPCyUUagZCqu4eqZRq?usp=sharing).

We imported the dataset we created directly from Tensorflow. To enable the training on Google Colab, instead of use the `data.yaml` file provided, we create a custom one `data-roboflow.yaml` (which can be found [here](https://drive.google.com/file/d/1Zo8eSazq72UtaSY6LfhtgmZIoyaMP5vh/view) or in the Google folder).

More of that, we created a mini-script to save autonomously - in a file in the Google folder - the number of epochs for which our model obtained a satisfying result.

## Instructions to use model_train.ipynb to train a model
To be able to run the script, you have to:

* If it is the first time you run the script, you should run it on the default weights `yolov8n.pt` provided by Ultralytics.

* Change the variable `drive_folder`, which corresponds to the path - from your *MyDrive* - where you want to save the results and on which you have the custom `data.yaml` file.
