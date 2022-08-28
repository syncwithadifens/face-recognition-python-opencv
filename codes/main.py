import sys
from PyQt5.QtWidgets import QApplication, QPushButton, QLabel, QWidget, QGridLayout, QLineEdit, QSizePolicy, QMessageBox, QCheckBox
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt
import imutils
import cv2
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import numpy as np
import dlib
import face_recognition
import os
from mylayout import *
from pathlib import Path


class MainWindow(myLayout):
    def __init__(self):
        super().__init__()

        self.address = 0
        self.cap = None
        self.frame = []
        self.writer = None
        self.THRESHOLD = 0.6

        # load face detector
        protoPath = str(Path().resolve().parent / 'model' / 'deploy.prototxt')
        modelPath = str(Path().resolve().parent / 'model' /
                        'res10_300x300_ssd_iter_140000.caffemodel')
        self.detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
        predictor = dlib.shape_predictor(
            str(Path().resolve().parent / 'model' / 'shape_predictor_68_face_landmarks.dat'))
        self.fa = FaceAligner(predictor, desiredFaceWidth=256)

        # load known faces
        self.loadFaces()

    def viewCam(self):
        ret, self.frame = self.cap.read()
        self.frame = imutils.resize(self.frame, width=self.width)
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        h, w, c = self.frame.shape
        step = c * w
        qImg = QImage(self.frame.data, w, h, step, QImage.Format_RGB888)
        self.image.setPixmap(QPixmap.fromImage(qImg))

    def resizeVideo(self):
        try:
            self.width = int(self.resize.text())
        except:
            QMessageBox.information(
                self, "QMessageBox.information()", 'start camera first!')

    def snapshot(self):
        try:
            filename = self.shotName.text()
            if not filename:
                filename = 'snapshot'
            filePath = Path().resolve().parent / 'output' / f'{filename}.jpg'
            cv2.imwrite(str(filePath), cv2.cvtColor(
                self.frame, cv2.COLOR_RGB2BGR))
            QMessageBox.information(
                self, "QMessageBox.information()", f'image is saved as: {filename}.jpg')
        except:
            QMessageBox.information(
                self, "QMessageBox.information()", 'start camera first!')

    def recording(self):
        self.videoName = self.recordName.text()
        if not self.videoName:
            self.videoName = 'video'
        # initialize video writer
        if self.writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            filePath = Path().resolve().parent / 'output' / \
                f'{self.videoName}.mp4'
            self.writer = cv2.VideoWriter(
                str(filePath), fourcc, 10, (self.frame.shape[1], self.frame.shape[0]), True)
        if self.writer is not None:
            self.writer.write(cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR))

    def addFace(self):
        if not self.cap and len(self.frame) == 0:
            QMessageBox.information(
                self, "QMessageBox.information()", 'start camera first!')
        elif self.detectTimer.isActive():
            QMessageBox.information(
                self, "QMessageBox.information()", 'please stop face recognition process!')
        else:
            filename = self.faceName.text()
            if not filename:
                filename = 'afiv'
            filePath = Path().resolve().parent / 'faces' / f'{filename}.jpg'
            cv2.imwrite(str(filePath), cv2.cvtColor(
                self.frame, cv2.COLOR_RGB2BGR))
            QMessageBox.information(
                self, "QMessageBox.information()", f'{filename} is added to database')

            # load image
            self.frame = cv2.imread(str(filePath))
            self.frame = imutils.resize(self.frame, width=600)
            (h, w) = self.frame.shape[:2]

            # construct a blob from the image
            imageBlob = cv2.dnn.blobFromImage(cv2.resize(
                self.frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)

            # detect face
            self.detector.setInput(imageBlob)
            detections = self.detector.forward()

            # loop over the detections
            for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with
                # the prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections
                if confidence > 0.6:

                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (self.startX, self.startY, self.endX,
                     self.endY) = box.astype("int")

                    # face alignment
                    face_aligned = self.alignedFace()
                    (fH, fW) = face_aligned.shape[:2]

                    # ensure the face width and height are sufficiently large
                    if fW < 20 or fH < 20:
                        continue

                    # encoding
                    try:
                        face_encoding = face_recognition.face_encodings(face_aligned)[
                            0]
                        filePath = Path().resolve().parent / \
                            'embeddings' / f'{filename}'
                        np.save(str(filePath), face_encoding)
                    except:
                        continue
            self.loadFaces()

    def loadFaces(self):
        filenames = os.listdir(Path().resolve().parent / 'embeddings')
        self.knownNames, self.knownEmbeddings = [], []
        for filename in filenames:
            self.knownNames.append(filename.split('.')[0])
            filePath = Path().resolve().parent / 'embeddings' / f'{filename}'
            self.knownEmbeddings.append(np.load(str(filePath)))

        self.box_colors = {}
        for name in self.knownNames:
            self.box_colors.update({name: (np.random.randint(
                128, 255), np.random.randint(128, 255), np.random.randint(128, 255))})
        self.box_colors.update({'siapa?': (255, 0, 0)})

    def alignedFace(self):
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        dlibrect = dlib.rectangle(int(self.startX), int(
            self.startY), int(self.endX), int(self.endY))
        face_aligned = self.fa.align(self.frame, gray, dlibrect)
        face_aligned = cv2.cvtColor(face_aligned, cv2.COLOR_BGR2RGB)
        # face_aligned = face_aligned[64:192,64:192]
        return face_aligned[32:224, 32:224]

    def detection(self):
        ret, self.frame_ori = self.cap.read()
        self.frame = self.frame_ori.copy()
        self.frame_ori = imutils.resize(self.frame_ori, width=self.width)
        self.frame = imutils.resize(self.frame, width=600)
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        self.frame_ori = cv2.cvtColor(self.frame_ori, cv2.COLOR_BGR2RGB)
        (h, w) = self.frame.shape[:2]
        (h_ori, w_ori) = self.frame_ori.shape[:2]
        rh, rw = h_ori/h, w_ori/w

        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(cv2.resize(
            self.frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # face detector to localize faces in the input image
        self.detector.setInput(imageBlob)
        detections = self.detector.forward()

        if len(detections) > 0:
            # loop over the detections
            for i in range(0, detections.shape[2]):
                # extract the confidence
                confidence = detections[0, 0, i, 2]
                # filter out weak detections
                if confidence > 0.5:
                    # get bounding box
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (self.startX, self.startY, self.endX,
                     self.endY) = box.astype("int")

                    # face alignment
                    face_aligned = self.alignedFace()
                    (fH, fW) = face_aligned.shape[:2]

                    # ensure the face width and height are sufficiently large
                    if fW < 20 or fH < 20:
                        continue

                    # encoding
                    try:
                        face_encoding = face_recognition.face_encodings(face_aligned)[
                            0]
                    except:
                        continue

                    # use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(
                        self.knownEmbeddings, face_encoding)
                    distance = np.min(face_distances)
                    if distance < self.THRESHOLD:
                        best_match_index = np.argmin(face_distances)
                        name = self.knownNames[best_match_index]
                    else:
                        name = 'siapa?'

                    # bounding box
                    sX, sY, eX, eY = int(
                        self.startX*rw), int(self.startY*rh), int(self.endX*rw), int(self.endY*rh)
                    cv2.rectangle(self.frame_ori, (sX, sY),
                                  (eX, eY), self.box_colors[name], 2)
                    # put the name
                    text = f'{name} {round(distance,2)}'
                    font_scale = .75
                    font = cv2.FONT_ITALIC
                    (text_width, text_height) = cv2.getTextSize(
                        text, font, fontScale=font_scale, thickness=1)[0]
                    # set the text start position
                    text_offset_x, text_offset_y = sX, sY - 10
                    # make the coords of the box with a small padding of two pixels
                    box_coords = ((text_offset_x - 5, text_offset_y + 5),
                                  (text_offset_x + text_width + 5, text_offset_y - text_height - 5))
                    overlay = self.frame_ori.copy()
                    cv2.rectangle(
                        overlay, box_coords[0], box_coords[1], self.box_colors[name], -1)
                    cv2.putText(overlay, text, (text_offset_x, text_offset_y),
                                font, fontScale=font_scale, color=(0, 0, 0), thickness=2)
                    # apply the overlay
                    alpha = 0.6
                    cv2.addWeighted(overlay, alpha, self.frame_ori,
                                    1 - alpha, 0, self.frame_ori)

        h, w, c = self.frame_ori.shape
        step = c * w
        qImg = QImage(self.frame_ori.data, w, h, step, QImage.Format_RGB888)
        self.image.setPixmap(QPixmap.fromImage(qImg))
        self.frame = self.frame_ori

    def controlTimer(self):
        # if timer is stopped
        if not self.timer.isActive():
            self.startBtn.setStyleSheet(
                "QPushButton {background-color: #DE3163}")
            self.cap = cv2.VideoCapture(self.address)
            self.timer.start(20)
            self.startBtn.setText("Stop Camera")
        # if timer is started
        else:
            if self.detectTimer.isActive():
                QMessageBox.information(
                    self, "QMessageBox.information()", 'please stop face recognition process!')
            else:
                self.startBtn.setStyleSheet(
                    "QPushButton {background-color: #f9e79f}")
                self.timer.stop()
                self.cap.release()
                self.cap = None
                self.startBtn.setText("Start Camera")

    def controlRecorder(self):
        # if timer is stopped
        if not self.recordTimer.isActive():
            if not self.cap:
                QMessageBox.information(
                    self, "QMessageBox.information()", 'start camera first!')
            else:
                self.recordBtn.setStyleSheet(
                    "QPushButton {background-color: #DE3163}")
                self.recordTimer.start(20)
                self.recordBtn.setText("Stop Recording")
        # if timer is started
        else:
            self.recordBtn.setStyleSheet(
                "QPushButton {background-color: #f9e79f}")
            self.recordTimer.stop()
            self.recordBtn.setText("Start Recording")
            self.writer = None
            QMessageBox.information(
                self, "QMessageBox.information()", f'video is saved as: {self.videoName}.mp4')

    def controlDetector(self):
        # if timer is stopped
        if not self.detectTimer.isActive():
            if not self.cap:
                QMessageBox.information(
                    self, "QMessageBox.information()", 'start camera first!')
            elif len(self.knownNames) == 0:
                QMessageBox.information(
                    self, "QMessageBox.information()", 'database is empty')
            else:
                self.detectBtn.setStyleSheet(
                    "QPushButton {background-color: #DE3163}")
                # start detection
                self.detectTimer.start(20)
                # stop viewCam
                self.timer.stop()
                self.detectBtn.setText("Stop Face Recognition")
        # if timer is started
        else:
            self.detectBtn.setStyleSheet(
                "QPushButton {background-color: #f9e79f}")
            # stop detection
            self.detectTimer.stop()
            # start viewCam
            self.timer.start(20)
            self.detectBtn.setText("Start Face Recognition")


if __name__ == '__main__':
    app = QApplication(sys.argv)

    mainWindow = MainWindow()
    mainWindow.show()

    sys.exit(app.exec_())
