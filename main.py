from datasetPrep import DatasetPrep
from datetime import datetime
from mlModel import MlModel
import os

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
