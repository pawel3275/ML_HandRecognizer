from datetime import datetime
from mlModel import MlModel
import os
import tensorflow as tf
from gui import Gui

dataset_directory = os.getcwd() + "\\dataset"
train_dataset_directory = dataset_directory + "\\train"
validation_dataset_directory = dataset_directory + "\\test"


def train_and_save_model():
    current_time = datetime.now()
    current_time = current_time.strftime("%d_%m_%Y_%H_%M_%S")
    model = MlModel(train_dataset_directory, validation_dataset_directory)
    model.create_aug_images()
    model.train_model()
    model.save_model("model_"+str(current_time))
    model.convert_model_to_light()


new_model = tf.keras.models.load_model("D:\\scratch\\ML_HandRecognizer\\ML_HandRecognizer\\models\\test")
new_model.summary()

window = Gui(new_model)
