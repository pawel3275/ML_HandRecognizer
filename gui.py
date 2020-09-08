import tkinter
import cv2
from PIL import Image, ImageTk
from mlModel import MlModel
import imutils
import numpy as np
import math
from sklearn.metrics import pairwise
import matplotlib.pyplot as plt

class Gui:
    def __init__(self, ml_model, video_device_id=0):
        self.mlModel = ml_model
        self.frame_counter = 0
        self.background = None
        self.wighted = None
        self.area_of_interest = (100, 100), (300, 300)

        self.window = tkinter.Tk()
        self.window.title("ML Hand Recognizer")
        self.video_device_id = video_device_id
        # self.window.minsize(480, 640)

        # open video source
        self.video_source = VideoCapture(self.video_device_id)

        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(self.window, width=self.video_source.width, height=self.video_source.height)
        self.canvas.pack()

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 1  # 15 for release
        self.update()

        self.window.mainloop()

    def update(self):
        # Get a frame from the video source
        status, frame = self.video_source.get_frame()
        if status:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

        frame = frame[50:450, 50:450]
        if self.frame_counter < 60:
            self.frame_counter += 1
            if self.background is None:
                self.background = frame.copy()
                self.background = np.float32(self.background)
            else:
                cv2.accumulateWeighted(frame.copy(), self.background, 0.2)
        else:
            back = cv2.convertScaleAbs(self.background)
            back_gray = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            img = cv2.absdiff(back_gray, frame_gray)

            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            con, hie = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            con = max(con, key=cv2.contourArea)
            conv_hull = cv2.convexHull(con)
            cv2.drawContours(img, [conv_hull], -1, 225, 3)

            circular_roi = np.zeros_like(img, dtype='uint8')
            wighted = cv2.addWeighted(img.copy(), 0.6, circular_roi, 0.4, 2)
            cv2.imshow('wighted', wighted)

        cv2.rectangle(frame, (50, 50), (400, 400), 255, 3)
        cv2.imshow('frame', frame)

        #image_to_predict, frame = Gui.preprocess_vdeo_frame(frame)
        #cv2.imshow("frame", frame)
        #cv2.imshow("image_to_predict", image_to_predict)
        #self.predict_from_frame(image_to_predict)
        self.window.after(self.delay, self.update)

    def predict_from_frame(self, image):
        image = cv2.resize(image, (32, 32))
        image = image / 255.0
        image = image[np.newaxis, :, :, np.newaxis]
        prediction = self.mlModel.predict(image)
        index = np.argmax(prediction)
        print(MlModel.classes_labels[index])

    @staticmethod
    def preprocess_vdeo_frame(frame):
        frame = cv2.flip(frame, 1)

        # Hand box
        hand_region = frame[100:450, 100:450]
        frame = cv2.rectangle(frame, (100, 100), (450, 450), (0, 255, 0), 0)

        hsv = cv2.cvtColor(hand_region, cv2.COLOR_BGR2HSV)

        # Skin tone color
        lower_skin_color = np.array([0, 48, 80], dtype=np.uint8)
        upper_skin_color = np.array([20, 255, 255], dtype=np.uint8)

        # define masks
        mask = cv2.inRange(hsv, lower_skin_color, upper_skin_color)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)

        return mask, frame


class VideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.video_frame = cv2.VideoCapture(video_source)
        if not self.video_frame.isOpened():
            print("Unable to open video source")

        # Get video source width and height
        self.width = self.video_frame.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.video_frame.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.video_frame.isOpened():
            status, frame = self.video_frame.read()
            if status:
                return status, frame
            else:
                return status, None

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.video_frame.isOpened():
            self.video_frame.release()
