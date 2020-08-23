import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from os import listdir, makedirs, path
from os.path import isfile, join
import os
import cv2

hands_numerical_labels = {
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

        size = ((len(self.train_files)), 64, 64, 3)
        self.array_with_train_images = np.zeros(size)
        size = ((len(self.train_files)), 1)
        self.array_with_train_labels = np.zeros(size)

        size = ((len(self.test_files)), 64, 64, 3)
        self.array_with_test_images = np.zeros(size)
        size = ((len(self.test_files)), 1)
        self.array_with_test_labels = np.zeros(size)

        self.accuracy = 0
        self.loss = 0

        self.model = None
        self.model_path = None

    def load_dataset(self):
        # I'm literally not proud of this function, this could be done better and be splitted,
        # but for now it will stay this way, it's just load data function after all
        counter = 0
        for file in self.train_files:
            # Load image to np array and process it to one dimensional array with pixel values
            image_as_array = cv2.imread(self.train_path + "\\" + file)

            # percent by which the image is resized
            scale_percent = 50

            # calculate the 50 percent of original dimensions
            width = int(image_as_array.shape[1] * scale_percent / 100)
            height = int(image_as_array.shape[0] * scale_percent / 100)
            image_as_array = cv2.resize(image_as_array, (width, height))

            image_as_array = image_as_array / 255.0

            # Get label from filename0
            image_label = file[file.rfind("_")+1:file.rfind(".")]
            image_label = hands_numerical_labels[image_label]

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

            # percent by which the image is resized
            scale_percent = 50

            # calculate the 50 percent of original dimensions
            width = int(image_as_array.shape[1] * scale_percent / 100)
            height = int(image_as_array.shape[0] * scale_percent / 100)
            image_as_array = cv2.resize(image_as_array, (width, height))

            image_as_array = image_as_array / 255.0

            # Get label from filename0
            image_label = file[file.rfind("_")+1:file.rfind(".")]
            image_label = hands_numerical_labels[image_label]

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
        self.model.add(layers.Conv2D(94, (3, 3), activation="relu", input_shape=(64, 64, 3)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation="relu"))
        self.model.add(layers.MaxPooling2D((2, 2)))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(125, activation="relu"))
        self.model.add(layers.Dense(56, activation="relu"))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Dense(12, activation="relu"))

        print(self.model.summary())

        self.model.compile(optimizer="adam",
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=["accuracy"])

        print("Model compiled successfully")

        self.model.fit(self.array_with_train_images,
                  self.array_with_train_labels,
                  epochs=3)

        self.loss, self.accuracy = self.model.evaluate(self.array_with_test_images, self.array_with_test_labels)
        self.accuracy = round(self.accuracy, 2) * 10
        self.loss = round(self.loss, 2) * 10

        print("Loss:", self.loss)
        print("Accuracy:", self.accuracy)

    def save_model(self, filename):
        directory = os.getcwd() + "\\models"
        if not path.exists(directory):
            makedirs(directory, exist_ok=True)

        self.model.save(directory + "\\" + filename + "_acc_" + str(self.accuracy))
        self.model_path = directory + "\\" + filename + "_acc_" + str(self.accuracy)

    def convert_model_to_light(self):
        converter = tf.lite.TFLiteConverter.from_saved_model(self.model_path)
        converted_model = converter.convert()
        open("{}.tflite".format(self.model_path), "wb").write(converted_model)


