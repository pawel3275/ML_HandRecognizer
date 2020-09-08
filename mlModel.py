import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from os import makedirs, path
import os
import cv2
from tensorflow.keras.callbacks import EarlyStopping


class MlModel:
    classes_labels = ["0", "1", "2", "3", "4", "5"]
    target_image_size = (32, 32)

    def __init__(self, train_path, test_path):
        tf.keras.backend.clear_session()

        self.train_path = train_path
        self.test_path = test_path

        self.model = None
        self.model_path = None
        self.train_gen = None
        self.test_gen = None

    @staticmethod
    def preprocess_image_for_model(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.GaussianBlur(image, (5, 5), 0)

        status, image = cv2.threshold(image, 80, 255, cv2.THRESH_BINARY)
        cv2.imshow("Image", image)
        image = cv2.resize(image, (32, 32))
        image = image / 255.0
        image = image[:, :, np.newaxis]

        return image

    @staticmethod
    def transform_dataset(dataset_path):
        for item in os.listdir(dataset_path):
            if os.path.isfile(dataset_path + "\\" + item):
                image = cv2.imread(dataset_path + "\\" + item)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.GaussianBlur(image, (5, 5), 0)

                status, image = cv2.threshold(image, 90, 255, cv2.THRESH_BINARY)
                cv2.imwrite(dataset_path + "\\" + item, image)

    def create_aug_images(self):
        batch_size = 128

        train_image_datagen = ImageDataGenerator(rescale=1./255,
                                                 rotation_range=60.,
                                                 width_shift_range=0.2,
                                                 height_shift_range=0.2,
                                                 zoom_range=0.3,
                                                 horizontal_flip=True,
                                                 vertical_flip=False)

        #test_datagen = ImageDataGenerator(rescale=1. / 255)

        test_datagen = ImageDataGenerator(rescale=1. / 255,
                                          rotation_range=60.,
                                          width_shift_range=0.2,
                                          height_shift_range=0.2,
                                          zoom_range=0.3,
                                          horizontal_flip=True,
                                          vertical_flip=False)

        self.train_gen = train_image_datagen.flow_from_directory(
            self.train_path,
            target_size=self.target_image_size,
            color_mode='grayscale',
            batch_size=batch_size,
            classes=MlModel.classes_labels,
            class_mode='categorical'
        )

        self.test_gen = test_datagen.flow_from_directory(
            self.test_path,
            target_size=self.target_image_size,
            color_mode='grayscale',
            batch_size=batch_size,
            classes=MlModel.classes_labels,
            class_mode='categorical'
        )

        X, y = self.train_gen.next()
        print(X.shape, y.shape)
        for i in range(50):
            name = "Image_"+str(i)+str(np.argmax(y[i]))+".png"
            img = np.uint8(255 * X[i, :, :, 0])
            cv2.imwrite(name, img)

    def train_model(self):
        self.model = models.Sequential()

        # Convolution layers
        self.model.add(layers.Conv2D(182, (3, 3), activation='relu', input_shape=(32, 32, 1)))
        self.model.add(layers.Conv2D(120, (3, 3), strides=(1, 1), activation='relu'))
        self.model.add(layers.Conv2D(92, (3, 3), strides=(1, 1), activation='relu'))
        self.model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same", activation='relu'))

        self.model.add(layers.Flatten())

        # Dense layers
        self.model.add(layers.Dense(182, activation='relu'))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Dense(86, activation='relu'))
        self.model.add(layers.Dense(32, activation='relu'))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Dense(6, activation='softmax', name='output_layer'))

        print(self.model.summary())

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

        print("Model compiled successfully")

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10)
        ]

        self.model.fit_generator(
            self.train_gen,
            steps_per_epoch=120,
            epochs=8,
            validation_data=self.test_gen,
            validation_steps=28,
            callbacks=callbacks
        )

    def save_model(self, filename):
        directory = os.getcwd() + "\\models"
        if not path.exists(directory):
            makedirs(directory, exist_ok=True)

        self.model.save(directory + "\\" + filename)
        self.model_path = directory + "\\" + filename

    def convert_model_to_light(self):
        converter = tf.lite.TFLiteConverter.from_saved_model(self.model_path)
        converted_model = converter.convert()
        open("{}.tflite".format(self.model_path), "wb").write(converted_model)


