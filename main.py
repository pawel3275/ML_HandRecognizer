from datasetPrep import DatasetPrep
from datetime import datetime
from mlModel import MlModel
import os
from gui import Gui
import tensorflow as tf

dataset_directory = os.getcwd() + "\\dataset"
train_dataset_directory = dataset_directory + "\\train"
validation_dataset_directory = dataset_directory + "\\test"

#DatasetPrep.download_dataset("https://www.kaggle.com/koryakinp/fingers/download", dataset_directory)

current_time = datetime.now()
current_time = current_time.strftime("%d_%m_%Y_%H_%M_%S")
model = MlModel(train_dataset_directory, validation_dataset_directory)
model.load_dataset()
model.train_model()
model.save_model("model_"+str(current_time))
model.convert_model_to_light()

new_model = tf.keras.models.load_model("D:\\scratch\\ML_HandRecognizer\\ML_HandRecognizer\\models\\test")
new_model.summary()

window = Gui(new_model)
