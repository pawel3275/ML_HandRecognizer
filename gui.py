import tkinter
import cv2
from PIL import Image, ImageTk
from mlModel import MlModel
import imutils
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
        self.delay = 15  # 15 for release
        self.update()

        self.window.mainloop()

    def update(self):
        # Get a frame from the video source
        status, frame = self.video_frame.get_frame()

        if status:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

        self.predict_from_frame(frame)
        self.window.after(self.delay, self.update)

    def predict_from_frame(self, image):
        image = MlModel.preprocess_image_for_model(image)
        image = image[np.newaxis, :, :, :]
        prediction = self.mlModel.predict(image)
        index = np.argmax(prediction)
        print(MlModel.classes_labels[index])


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

    @staticmethod
    def get_hand_countour(image):
        # threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh = cv2.threshold(image, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # find contours in thresholded image, then grab the largest one
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        print(type(cnts))
        if not cnts:
            return
        c = max(cnts, key=cv2.contourArea)

        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        cv2.drawContours(image, [c], -1, (0, 255, 255), 2)
        cv2.circle(image, extLeft, 8, (0, 0, 255), -1)
        cv2.circle(image, extRight, 8, (0, 255, 0), -1)
        cv2.circle(image, extTop, 8, (255, 0, 0), -1)
        cv2.circle(image, extBot, 8, (255, 255, 0), -1)

        # show the output image
        cv2.imshow("Image", image)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.video_frame.isOpened():
            self.video_frame.release()
