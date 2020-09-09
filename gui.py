import numpy as np
import tkinter
import cv2
from PIL import Image, ImageTk
from mlModel import MlModel


class Gui:
    def __init__(self, ml_model, video_device_id=0):
        self.mlModel = ml_model
        self.frame_counter = 0
        self.background = None
        self.image_to_predict = None
        self.prediction = 0

        self.window = tkinter.Tk()
        self.window.title("ML Hand Recognizer")
        self.video_device_id = video_device_id

        # open video source
        self.video_source = VideoCapture(self.video_device_id)
        self.window.minsize(int(self.video_source.width/2), int(self.video_source.height/2))
        self.window.maxsize(int(self.video_source.width), int(self.video_source.height))

        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(self.window, width=self.video_source.width, height=self.video_source.height)
        self.canvas.pack()

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15  # 15 for release
        self.update()

        self.window.mainloop()

    def update(self):
        # Get a frame from the video source
        status, frame = self.video_source.get_frame()
        if status:
            show_frame = frame
            cv2.rectangle(show_frame, (50, 50), (450, 450), 255, 3)
            show_frame = cv2.putText(show_frame, str(self.prediction), (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                (0, 255, 0), 2, cv2.LINE_AA, False)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(show_frame))
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

            circular_roi = np.zeros_like(img, dtype='uint8')
            self.image_to_predict = cv2.addWeighted(img.copy(), 0.6, circular_roi, 0.4, 2)
            cv2.imshow('image_to_predict', self.image_to_predict)

        self.predict_from_frame(self.image_to_predict)
        self.window.after(self.delay, self.update)

    def predict_from_frame(self, image):
        if image is None:
            return
        image = cv2.GaussianBlur(image, (5, 5), 0)
        image = cv2.resize(image, (32, 32))
        image = image / 255.0
        image = image[np.newaxis, :, :, np.newaxis]

        prediction = self.mlModel.predict(image)
        index = np.argmax(prediction)
        self.prediction = MlModel.classes_labels[index]
        print(self.prediction)


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
