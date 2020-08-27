import tensorflow as tf
from tensorflow.keras import layers, models, Input
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

        size = ((len(self.train_files)), 64, 64, 1)
        self.array_with_train_images = np.zeros(size)
        size = ((len(self.train_files)), 1)
        self.array_with_train_labels = np.zeros(size)

        size = ((len(self.test_files)), 64, 64, 1)
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

            image_as_array = self.preprocess_image_for_model(image_as_array)

            # Get label from filename0
            image_label = file[file.rfind("_")+1:file.rfind(".")]
            image_label = hands_numerical_labels[image_label]

            # Append data to lists
            self.array_with_train_images[counter][:] = image_as_array
            self.array_with_train_labels[counter][:] = image_label

            counter += 1

        print("#############################################")
        print("Loading and processing train files finished")
        print("Shape of training dataset: ", self.array_with_train_images.shape)
        print("Ndim of train array: ", self.array_with_train_images.ndim)
        print("Shape of training labels: ", self.array_with_train_labels.shape)
        print("Ndim of training labels: ", self.array_with_train_labels.ndim)
        print("Ndim of first image: ", self.array_with_train_images[0].ndim)
        print("Shape of first image: ", self.array_with_train_images[0].shape)
        print("#############################################")

        counter = 0
        for file in self.test_files:
            # Load image to np array and process it to one dimensional array with pixel values
            image_as_array = cv2.imread(self.test_path + "\\" + file)

            image_as_array = self.preprocess_image_for_model(image_as_array)

            # Get label from filename0
            image_label = file[file.rfind("_")+1:file.rfind(".")]
            image_label = hands_numerical_labels[image_label]

            # Append data to lists
            self.array_with_test_images[counter][:] = image_as_array
            self.array_with_test_labels[counter][:] = image_label

            counter += 1

        print("#############################################")
        print("Loading and processing test files finished")
        print("Shape of test dataset: ", self.array_with_test_images.shape)
        print("Ndim of test array: ", self.array_with_test_images.ndim)
        print("Shape of test labels: ", self.array_with_test_labels.shape)
        print("Ndim of test labels: ", self.array_with_test_labels.ndim)
        print("Ndim of first image: ", self.array_with_test_images[0].ndim)
        print("Shape of first image: ", self.array_with_test_images[0].shape)
        print("#############################################")

        # This function could actually return x_test, y_test, y_train, x_train
        # but I'm assigning this to values of the object itself instead

    @staticmethod
    def preprocess_image_for_model(image):
        image = cv2.resize(image, (64, 64))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image / 255.0
        image = image[:, :, np.newaxis]

        return image

    def train_model(self):
        self.model = models.Sequential()
        self.model.add(Input(shape=(64, 64, 1), name="Input_layer"))
        self.model.add(layers.Conv2D(94, (3, 3), activation="relu", name="first_Conv2D"))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation="relu", name="second_Conv2D"))
        self.model.add(layers.MaxPooling2D((2, 2)))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(125, activation="relu", name="first_Dense"))
        self.model.add(layers.Dense(56, activation="relu", name="second_Dense"))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Dense(12, activation="relu", name="third_Dense"))

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


