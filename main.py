from datetime import datetime
from mlModel import MlModel
from gui import Gui
import tensorflow as tf
import pathlib
import os


dataset_directory = os.getcwd() + "\\dataset"
train_dataset_directory = dataset_directory + "\\train"
validation_dataset_directory = dataset_directory + "\\test"

current_time = datetime.now()
current_time = current_time.strftime("%d_%m_%Y_%H_%M_%S")
model_path = str(pathlib.Path().absolute()) + "\\models\\model_"+str(current_time)


def train_and_save_model():
    model = MlModel(train_dataset_directory, validation_dataset_directory)
    model.create_aug_images()
    model.train_model()
    model.save_model("model_"+str(current_time))
    model.convert_model_to_light()


def show_gui_and_load_model():
    print(model_path)
    new_model = tf.keras.models.load_model(model_path)
    new_model.summary()
    window = Gui(new_model)


if __name__ == "__main__":
    train_and_save_model()
    show_gui_and_load_model()



