import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from os import listdir, makedirs, path
from os.path import isfile, join
import os
import cv2

hands_iteral_labels = {
    "0L": 0,
    "1L": 1,
    "2L": 2,
    "3L": 3,
    "4L": 4,
    "5L": 5,
    "0R": 6,
    "1R": 7,
    "2R": 8,
    "3R": 9,
    "4R": 10,
    "5R": 11
}


class MlModel:
    def __init__(self, train_path, test_path):
        tf.keras.backend.clear_session()
        self.train_path = train_path
        self.test_path = test_path

        self.train_files = [file for file in listdir(train_path) if isfile(join(train_path, file))]
        self.test_files = [file for file in listdir(test_path) if isfile(join(test_path, file))]

        size = ((len(self.train_files)), 128, 128, 3)
        self.array_with_train_images = np.zeros(size)
        size = ((len(self.train_files)), 1)
        self.array_with_train_labels = np.zeros(size)

        size = ((len(self.test_files)), 128, 128, 3)
        self.array_with_test_images = np.zeros(size)
        size = ((len(self.test_files)), 1)
        self.array_with_test_labels = np.zeros(size)

    def load_dataset(self):
        # I'm literally not proud of this function, this could be done better and be splitted,
        # but for now it will stay this way, it's just load data function after all
        counter = 0
        for file in self.train_files:
            # Load image to np array and process it to one dimensional array with pixel values
            image_as_array = cv2.imread(self.train_path + "\\" + file)
            image_as_array = image_as_array / 255.0
            #image_as_array = image_as_array.reshape(1, 128*128*3)

            # Get label from filename0
            image_label = file[file.rfind("_")+1:file.rfind(".")]
            image_label = hands_iteral_labels[image_label]

            # Append data to lists
            self.array_with_train_images[counter][:] = image_as_array
            self.array_with_train_labels[counter][:] = image_label

            counter += 1

        print("Loading and processing train files finished")
        print("Shape of training dataset: ", self.array_with_train_images.shape)
        print("Example of first image:", self.array_with_train_images[0])
        print("Shape of training labels: ", self.array_with_train_labels.shape)

        counter = 0
        for file in self.test_files:
            # Load image to np array and process it to one dimensional array with pixel values
            image_as_array = cv2.imread(self.test_path + "\\" + file)
            image_as_array = image_as_array / 255.0
            #image_as_array = image_as_array.reshape(1, 128*128*3)

            # Get label from filename0
            image_label = file[file.rfind("_")+1:file.rfind(".")]
            image_label = hands_iteral_labels[image_label]

            # Append data to lists
            self.array_with_test_images[counter][:] = image_as_array
            self.array_with_test_labels[counter][:] = image_label

            counter += 1

        print("Loading and processing test files finished")
        print("Shape of test dataset: ", self.array_with_test_images.shape)
        print("Example of first image:", self.array_with_test_images[0])
        print("Shape of test labels: ", self.array_with_test_labels.shape)

        # This function could actually return x_test, y_test, y_train, x_train
        # but I'm assigning this to values of the object itself instead

    def train_model(self):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(94, (3, 3), activation="relu", input_shape=(128, 128, 3)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(72, (3, 3), activation="relu"))
        self.model.add(layers.Conv2D(64, (3, 3), activation="relu"))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(260, activation="relu"))
        self.model.add(layers.Dense(142, activation="relu"))
        self.model.add(layers.Dropout(0.3))
        self.model.add(layers.Dense(56, activation="relu"))
        self.model.add(layers.Dense(12, activation="relu"))

        print(self.model.summary())

        self.model.compile(optimizer="adam",
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=["accuracy"])

        print("Model compiled successfully")

        self.model.fit(self.array_with_train_images,
                  self.array_with_train_labels,
                  epochs=2)

        self.loss, self.accuracy = self.model.evaluate(self.array_with_test_images, self.array_with_test_labels)

        print("Loss:", self.loss)
        print("Accuracy:", self.accuracy)

    def save_model(self, filename):
        directory = os.getcwd() + "\\models"
        if not path.exists(directory):
            makedirs(directory, exist_ok=True)

        self.model.save(directory + "\\" + filename + "_acc_" + str(self.accuracy))





