from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers, models
from os import makedirs, path
import tensorflow as tf
import numpy as np
import os
import cv2



class MlModel:
    labels = ["0", "1", "2", "3", "4", "5"]
    target_image_size = (32, 32)

    def __init__(self, train_path, test_path):
        '''
        Constructor for the ML model
        :param train_path: path in which labeled train images are present.
        :param test_path: path in which labeled test images are present.
        '''
        tf.keras.backend.clear_session()

        self.train_path = train_path
        self.test_path = test_path

        self.model = None
        self.model_path = None
        self.train_gen = None
        self.test_gen = None

    @staticmethod
    def process_image_for_model(image, image_size):
        '''
        Process image to specific image size. Operations which are performed are:
         - Color to gray,
         - Gaussian Blur of the image,
         - applied threshold,
         - resizing image to given size,
         - chagining image pixel values to be float values in range 0..1,
         - appending new axis for the depth at the end of an image array.
        :param image: image to perform the processing on.
        :param image_size: target size of an image as a tuple, for example: (32, 32)
        :return: processed image
        '''
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.GaussianBlur(image, (5, 5), 0)

        status, image = cv2.threshold(image, 80, 255, cv2.THRESH_BINARY)
        image = cv2.resize(image, image_size)
        image = image / 255.0
        image = image[:, :, np.newaxis]

        return image

    def create_aug_images(self):
        '''
        Create additional images for better model training process. Since input from the camera is not perfect,
        we need to perform additional rotations with extending the hand itself. this will make convolutional network
        to perform better as the filter will be more general in recognizing hands features from the input.
        :return: None.
        '''
        # Batch size, this can be changed to lower or higher number depending on gpu used. For lower performance gpu
        # Use lower value for batch size.
        batch_size = 128

        # Perform data generation of images for network
        train_image_datagen = ImageDataGenerator(rescale=1./255,  # To work on values from range 0..1
                                                 rotation_range=60.,  # 60 for simplicity, max 180 degrees
                                                 width_shift_range=0.2,  # extend width shape of image
                                                 height_shift_range=0.3,  # extend height shape of image
                                                 zoom_range=0.1,  # self explanatory
                                                 horizontal_flip=True,  # hand can be flipped horizontally
                                                 vertical_flip=False)  # hand can not be flipped vertically

        test_datagen = ImageDataGenerator(rescale=1. / 255,  # To work on values from range 0..1
                                          rotation_range=60.,  # 60 for simplicity, max 180 degrees
                                          width_shift_range=0.2,  # extend width shape of image
                                          height_shift_range=0.3,  # extend height shape of image
                                          zoom_range=0.1,  # self explanatory
                                          horizontal_flip=True,  # hand can be flipped horizontally
                                          vertical_flip=False)  # hand can not be flipped vertically

        # Perform required transitions of the image specified above on both train and test dataset
        self.train_gen = train_image_datagen.flow_from_directory(
            self.train_path,
            target_size=self.target_image_size,
            color_mode="grayscale",
            batch_size=batch_size,
            classes=MlModel.labels,
            class_mode="categorical"
        )

        self.test_gen = test_datagen.flow_from_directory(
            self.test_path,
            target_size=self.target_image_size,
            color_mode="grayscale",
            batch_size=batch_size,
            classes=MlModel.labels,
            class_mode="categorical"
        )

    def train_model(self):
        '''
        Compile and train model
        :return: None
        '''
        self.model = models.Sequential()

        # Convolution layers
        self.model.add(layers.Conv2D(182, (3, 3), activation="relu", input_shape=(32, 32, 1)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(120, (3, 3), strides=(1, 1), activation="relu"))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(92, (3, 3), strides=(1, 1), activation="relu"))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same", activation="relu"))

        self.model.add(layers.Flatten())

        # Dense layers
        self.model.add(layers.Dense(182, activation="relu"))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Dense(86, activation="relu"))
        self.model.add(layers.Dense(32, activation="relu"))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Dense(6, activation="softmax", name="output_layer"))

        print(self.model.summary())

        # Compile model.
        # Use caterogical crossentropy as this is simple categorization of the image problem.
        # Metrics acc for accuracy.
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])

        print("Model compiled successfully")

        # Define callback for early stopping when growth rate of accuracy is no longer occuring.
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=10)
        ]

        # Perform fitting dataset to model with accuracy measurement for test and train dataset
        self.model.fit_generator(
            self.train_gen,
            steps_per_epoch=120,
            epochs=8,
            validation_data=self.test_gen,
            validation_steps=28,
            callbacks=callbacks
        )

    def save_model(self, filename):
        '''
        Saves model in current directory with specified filename.
        :param filename: name of the model to be saved in "models" directory
        :return: None
        '''
        directory = os.getcwd() + "\\models"
        if not path.exists(directory):
            makedirs(directory, exist_ok=True)

        self.model.save(directory + "\\" + filename)
        self.model_path = directory + "\\" + filename

    def convert_model_to_light(self):
        '''
        Converts already saved model to the light version of the model which was already
        generated during train up session. Model from which light one is constructed is taken from
        already pre existing model from "models" directory.
        :return: None
        '''
        converter = tf.lite.TFLiteConverter.from_saved_model(self.model_path)
        converted_model = converter.convert()
        open("{}.tflite".format(self.model_path), "wb").write(converted_model)
