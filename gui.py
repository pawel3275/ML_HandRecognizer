import tkinter
import cv2
from PIL import Image, ImageTk
from mlModel import MlModel
import tensorflow as tf
import numpy as np


class Gui:
    def __init__(self, ml_model, video_source=0):
        self.mlModel = ml_model
        self.photo = None
        self.window = tkinter.Tk()
        self.window.title("ML Hand Recognizer")
        self.video_source = video_source
        # self.window.minsize(480, 640)

        # open video source
        self.video_frame = VideoCapture(self.video_source)

        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(self.window, width=self.video_frame.width, height=self.video_frame.height)
        self.canvas.pack()

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 150  # 15 for release
        self.update()

        self.window.mainloop()

    def update(self):
        # Get a frame from the video source
        status, frame, subtracted_frame = self.video_frame.get_subtracted_frame()

        if status:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

        self.predict_from_frame(subtracted_frame)
        self.window.after(self.delay, self.update)

    def predict_from_frame(self, image):
        image = MlModel.preprocess_image_for_model(image)
        image = image[np.newaxis, :, :, :]
        print(tf.sigmoid(self.mlModel.predict(image)))


class VideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.video_frame = cv2.VideoCapture(video_source)
        if not self.video_frame.isOpened():
            print("Unable to open video source")

        # Get video source width and height
        self.width = self.video_frame.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.video_frame.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_subtracted_frame(self):
        if self.video_frame.isOpened():
            status, frame = self.video_frame.read()
            if status:
                size = 200
                top_left_corner = (int(frame.shape[1] / 2) - size, int(frame.shape[0] / 2) - size)
                bottom_right_corner = (int(frame.shape[1] / 2) + size, int(frame.shape[0] / 2) + size)
                subtracted_frame = cv2.rectangle(frame, top_left_corner, bottom_right_corner, (255, 0, 0), 4)
                return status, frame, subtracted_frame
            else:
                return status, None

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.video_frame.isOpened():
            self.video_frame.release()
