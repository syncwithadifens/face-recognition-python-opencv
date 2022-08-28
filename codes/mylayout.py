from PyQt5.QtWidgets import QApplication, QPushButton, QLabel, QWidget, QGridLayout, QLineEdit, QSizePolicy, QVBoxLayout
from PyQt5.QtWidgets import QMessageBox, QCheckBox, QSlider, QLCDNumber, QFileSystemModel, QTreeView
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt
import imutils
import cv2
from imutils.face_utils import FaceAligner
from pop_faces import *
import platform


class myLayout(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f'Project UAS Kelompok 5')
        self.resize(700, 400)
        self.width = 900

        self.setStyleSheet('''
            QWidget{
                background-color: black;
            }
            QPushButton:pressed{
                background-color: black;
            }
            QPushButton{
                background-color: #f9e79f;
                border-style: outset;
                border-width: 2px;
                border-radius: 10px;
                border-color: beige;
                font: bold 14px;
                min-width: 10em;
                padding: 6px;
            }
            QLineEdit{
                background-color: #abebc6;
                border-style: outset;
                border-width: 2px;
                border-radius: 10px;
                border-color: beige;
                font: 14px;
                min-width: 5em;
                padding: 6px;
            }
            QLabel{
                color: white;
                border-width: 1px;
                border-radius: 20px;
                border-color: beige;
                font: bold 14px;
                padding: 6px;
            }
            QLCDNumber{
                background-color: #abebc6;
                border-style: outset;
                border-width: 2px;
                border-radius: 10px;
                border-color: beige;
                min-width: 5em;
                padding: 6px;
            }
        ''')

        # grid widgets 1
        w1 = QWidget()
        w1.resize(w1.sizeHint())
        w1.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.gridBtns = QGridLayout(w1)

        # ----------- CAMERA -----------

        simpleStream = QLabel(' ')
        simpleStream.setFont(QFont("Times", 32, QFont.Bold))
        simpleStream.setAlignment(Qt.AlignCenter)

        # start/stop button
        self.startBtn = QPushButton('Start Camera')
        sourceLabel = QLabel('Source:')
        sourceLabel.setAlignment(Qt.AlignRight)
        source = QLineEdit()
        source.setPlaceholderText('0')

        # show image
        self.timer = QTimer()
        self.timer.timeout.connect(self.viewCam)
        self.startBtn.clicked.connect(self.controlTimer)

        # resize video button
        resizeBtn = QPushButton('Resize Video')
        resizeBtn.clicked.connect(self.resizeVideo)
        resizeLabel = QLabel('Width:')
        resizeLabel.setAlignment(Qt.AlignRight)
        self.resize = QLineEdit()
        self.resize.setPlaceholderText(str(self.width))

        # snapshot button
        shotBtn = QPushButton('Snapshot')
        shotBtn.clicked.connect(self.snapshot)
        shotLabel = QLabel('Save As:')
        shotLabel.setAlignment(Qt.AlignRight)
        self.shotName = QLineEdit()
        self.shotName.setPlaceholderText('snapshot')

        # record button
        self.recordBtn = QPushButton('Start Recording')
        recordLabel = QLabel('Save As:')
        recordLabel.setAlignment(Qt.AlignRight)
        self.recordName = QLineEdit()
        self.recordName.setPlaceholderText('video')

        # record process
        self.recordTimer = QTimer()
        self.recordTimer.timeout.connect(self.recording)
        self.recordBtn.clicked.connect(self.controlRecorder)

        # output media button
        self.mediaBtn = QPushButton('Output Media Files')
        self.mediaBtn.clicked.connect(self.launchViewFolder)

        # add widgets to the grid
        self.gridBtns.addWidget(simpleStream, 0, 0, 1, 3)
        self.gridBtns.addWidget(self.startBtn, 1, 0)
        self.gridBtns.addWidget(sourceLabel, 1, 1)
        self.gridBtns.addWidget(source, 1, 2)
        self.gridBtns.addWidget(resizeBtn, 2, 0)
        self.gridBtns.addWidget(resizeLabel, 2, 1)
        self.gridBtns.addWidget(self.resize, 2, 2)
        self.gridBtns.addWidget(shotBtn, 3, 0)
        self.gridBtns.addWidget(shotLabel, 3, 1)
        self.gridBtns.addWidget(self.shotName, 3, 2)
        self.gridBtns.addWidget(self.recordBtn, 4, 0)
        self.gridBtns.addWidget(recordLabel, 4, 1)
        self.gridBtns.addWidget(self.recordName, 4, 2)
        self.gridBtns.addWidget(self.mediaBtn, 5, 0, 1, 3)

        # ----------- CAMERA -----------

        # ----------- FACE RECOGNITION -----------

        faceRecog = QLabel(' ')
        faceRecog.setAlignment(Qt.AlignCenter)
        faceIdx = cv2.imread(
            str(Path().resolve().parent / 'images' / 'facial.png'))
        faceIdx = imutils.resize(faceIdx, width=120)
        faceIdx = cv2.cvtColor(faceIdx, cv2.COLOR_BGR2RGB)
        height, width, channel = faceIdx.shape
        step = channel * width
        qImg = QImage(faceIdx.data, width, height, step, QImage.Format_RGB888)
        faceRecog.setPixmap(QPixmap.fromImage(qImg))

        # detection button
        self.detectBtn = QPushButton('Start Face Recognition')

        # detection and recognition
        self.detectTimer = QTimer()
        self.detectTimer.timeout.connect(self.detection)
        self.detectBtn.clicked.connect(self.controlDetector)

        # add face to database
        addfaceBtn = QPushButton('Add Face')
        addfaceBtn.clicked.connect(self.addFace)
        addfaceLabel = QLabel('Name:')
        addfaceLabel.setAlignment(Qt.AlignRight)
        self.faceName = QLineEdit()
        self.faceName.setPlaceholderText('afiv')

        # known faces
        self.showFacesBtn = QPushButton('Show Known Faces')
        self.showFacesBtn.clicked.connect(self.launchPopup)

        # threshold control
        thres = QLabel('Threshold:')
        thres.setAlignment(Qt.AlignRight)
        self.lcd = QLCDNumber()
        self.lcd.display(0.6)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setValue(60)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.valueChanged.connect(self.update_threshold)

        # add widgets to the grid
        self.gridBtns.addWidget(faceRecog, 6, 0, 1, 3)
        self.gridBtns.addWidget(self.detectBtn, 7, 0, 1, 3)
        self.gridBtns.addWidget(addfaceBtn, 8, 0)
        self.gridBtns.addWidget(addfaceLabel, 8, 1)
        self.gridBtns.addWidget(self.faceName, 8, 2)
        self.gridBtns.addWidget(self.showFacesBtn, 9, 0, 1, 3)
        self.gridBtns.addWidget(thres, 10, 0)
        self.gridBtns.addWidget(self.lcd, 10, 1, 3, 2)
        self.gridBtns.addWidget(self.slider, 13, 0, 1, 3)

        # ----------- FACE RECOGNITION -----------

        # image/video
        self.image = QLabel()
        imgIdx = cv2.imread(
            str(Path().resolve().parent / 'images' / 'index.png'))
        imgIdx = imutils.resize(imgIdx, width=self.width)
        imgIdx = cv2.cvtColor(imgIdx, cv2.COLOR_BGR2RGB)
        height, width, channel = imgIdx.shape
        step = channel * width
        qImg = QImage(imgIdx.data, width, height, step, QImage.Format_RGB888)
        self.image.setPixmap(QPixmap.fromImage(qImg))

        # whole layout
        layout = QGridLayout()
        layout.addWidget(w1, 0, 0)
        layout.addWidget(self.image, 0, 1)
        self.setLayout(layout)

    def update_threshold(self):
        x = self.slider.value()
        self.THRESHOLD = x/100
        self.lcd.display(x/100)

    def launchPopup(self):
        if len(self.knownNames) == 0:
            QMessageBox.information(
                self, "QMessageBox.information()", 'database is empty')
        elif self.detectTimer.isActive():
            QMessageBox.information(
                self, "QMessageBox.information()", 'please stop face recognition process!')
        else:
            pop = listFaces(self)
            pop.show()

    def launchViewFolder(self):
        dirPath = Path().resolve().parent / 'output'
        list_files = os.listdir(dirPath)
        list_files = [i.split('.')[0] for i in list_files if i.split('.')[
            1] in ['jpg', 'mp4']]
        if len(list_files) == 0:
            QMessageBox.information(
                self, "QMessageBox.information()", 'folder is empty')
        else:
            pop = viewFolder(self, str(dirPath))
            pop.show()


class viewFolder(QDialog):
    def __init__(self, parent, path):
        super().__init__(parent)
        self.setStyleSheet('''
        QWidget{
            background-color: #34495e;
            font: bold 12px;
        }
        QTreeView{
            background-color: white;
            border-style: outset;
            border-width: 2px;
            border-radius: 10px;
            border-color: beige;
            font: 14px;
            padding: 6px;
            width: 4px;
        }
        QHeaderView{
            background-color: white;
            font: bold 14px;
        }
        QPushButton:pressed{
            background-color: #34495e;
        }
        QPushButton{
            background-color: #6495ED;
            border-style: outset;
            border-width: 2px;
            border-radius: 10px;
            border-color: beige;
            font: bold 14px;
            min-width: 12em;
            padding: 6px;
            width: 4px;
        }
        QLabel{
            color: white;
            border-width: 1px;
            border-radius: 20px;
            border-color: beige;
            font: bold 20px;
            padding: 6px;
        }
        QLineEdit{
            background-color: white;
            border-style: outset;
            border-width: 2px;
            border-radius: 10px;
            border-color: beige;
            font: 14px;
            min-width: 6em;
            padding: 2px;
        }
        ''')
        w, h = 520, 400
        self.setWindowTitle('Output Media Files')
        self.setGeometry(300, 300, w, h)

        # Delete File
        delBtn = QPushButton('Delete Selected File')
        delBtn.clicked.connect(self.deleteFile)

        # Tree
        self.model = QFileSystemModel()
        self.model.setRootPath(path)

        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.tree.setRootIndex(self.model.index(path))
        self.tree.setColumnWidth(0, 160)
        self.tree.setColumnWidth(1, 100)
        self.tree.setColumnWidth(2, 80)
        self.tree.setAlternatingRowColors(True)

        self.tree.doubleClicked.connect(self.openFile)

        layout = QGridLayout()
        layout.addWidget(delBtn, 0, 0)
        layout.addWidget(self.tree, 1, 0, 1, 2)

        self.setLayout(layout)

    def openFile(self):
        idx = self.tree.currentIndex()
        filePath = self.model.filePath(idx)
        if platform.system() == 'Windows':
            os.startfile(filePath)
        elif platform.system() == 'Linux':
            os.system(f'xdg-open {filePath}')
        elif platform.system() == 'Darwin':
            os.system(f'open {filePath}')

    def deleteFile(self):
        idx = self.tree.currentIndex()
        filePath = self.model.filePath(idx)
        name = str(self.model.data(idx))
        if name != 'None':
            message = f'do you want to remove "{name}"?'
            reply = QMessageBox.question(self, 'Remove File', message,
                                         QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel)
            if reply == QMessageBox.Yes:
                filePath = self.model.filePath(idx)
                os.remove(filePath)
        else:
            QMessageBox.information(
                self, "QMessageBox.information()", 'please select a file!')
