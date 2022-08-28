from PyQt5.QtWidgets import QDialog, QLabel, QTableView, QPushButton, QGridLayout, QLineEdit, QWidget, QVBoxLayout, QMessageBox, QTableWidget
from PyQt5.QtCore import QAbstractTableModel, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.Qt import QTableWidgetItem, QAbstractItemView
import os
import cv2
import imutils
from pathlib import Path



class listFaces(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.resize(300, 400)
        self.setWindowTitle('Known Faces')

        self.setStyleSheet('''
        QWidget{
            background-color: #34495e;
            font: bold 12px;
        }
        QTableWidget{
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
            font: bold 12px;
            min-width: 4em;
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
            min-width: 5em;
            padding: 2px;
        }
        ''')

        layout = QGridLayout()
        knownNames = self.parent.knownNames.copy()
        knownNames.sort()

        self.rows = [[i] for i in knownNames]

        self.total = QLabel(f'Total Data: {len(knownNames)}')
  
        selectName = QLabel('Selected Name:')
        selectName.setStyleSheet("QLabel {font: 14px}")
        selectName.setAlignment(Qt.AlignRight)
        self.edit = QLineEdit()
        self.edit.setAlignment(Qt.AlignCenter)

        # Tabel Names
        self.createTable()

        # Buttons
        deleteBtn = QPushButton('Delete Name')
        deleteBtn.setStyleSheet('''
            QPushButton{background-color: #FF7F50}
            QPushButton:pressed{background-color: #34495e}
            ''')
        deleteBtn.clicked.connect(self.deleteName)
        showBtn = QPushButton('Show Image')
        showBtn.clicked.connect(self.showImage)

        # Add Widgets to Layout
        layout.addWidget(self.total, 0, 0)
        layout.addWidget(selectName, 1, 0)
        layout.addWidget(self.edit, 1, 1)
        layout.addWidget(deleteBtn, 2, 0)
        layout.addWidget(showBtn, 2, 1)
        layout.addWidget(self.view, 3, 0, 1, 2)

        self.setLayout(layout)


    def createTable(self):
        self.view = QTableWidget()
        self.view.setRowCount(len(self.rows))
        self.view.setColumnCount(len(self.rows[0]))
        self.view.setShowGrid(False)
        self.view.horizontalHeader().hide()
        self.view.verticalHeader().hide()
        self.view.clicked.connect(self.clickedTable)
  
        for row in enumerate(self.rows):
            for col in enumerate(row[1]):
                item = QTableWidgetItem()
                item.setText(col[1])
                self.view.setItem(row[0], col[0], item)


    def clickedTable(self):
        index = self.view.selectedIndexes()[0]
        val = self.view.model().data(index)
        self.edit.setText(val)


    def showImage(self):
        name = self.edit.text()
        list_images = os.listdir(Path().resolve().parent / 'faces')
        list_images = [i.split('.')[0] for i in list_images if i.split('.')[1]=='jpg']
        if name  == '':
            QMessageBox.information(self, "QMessageBox.information()", 'please select a name!')
        elif name not in list_images:
            QMessageBox.information(self, "QMessageBox.information()", f'{name} not found')
        else:
            pop = popupImage(name, self)
            pop.show()


    def deleteName(self):
        name = self.edit.text()
        list_images = os.listdir(Path().resolve().parent / 'faces')
        list_images = [i.split('.')[0] for i in list_images if i.split('.')[1]=='jpg']
        if name == '':
            QMessageBox.information(self, "QMessageBox.information()", 'please select a name!')
        elif name not in list_images:
            QMessageBox.information(self, "QMessageBox.information()", f'{name} not found')
        else:
            message = f'do you want to remove "{name}" from database?'
            reply = QMessageBox.question(self, 'Remove Name', message, \
                QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel)
            if reply == QMessageBox.Yes:        
                os.remove(Path().resolve().parent / 'faces' / f'{name}.jpg')
                os.remove(Path().resolve().parent / 'embeddings' / f'{name}.npy')
                self.parent.loadFaces()
                knownNames = self.parent.knownNames.copy()
                knownNames.sort()
                self.total.setText(f'Total Data: {len(knownNames)}')
                self.view.removeRow(self.view.currentRow())
                self.edit.setText('')

            





class popupImage(QDialog):
    def __init__(self, name, parent):
        super().__init__(parent)
        self.setWindowTitle(name)
        self.setStyleSheet('''
        QWidget{
            background-color: white;
        }
        ''')
        imageLabel = QLabel()
        layout = QVBoxLayout()
        image = cv2.imread(str(Path().resolve().parent / 'faces' / f'{name}.jpg'))
        image = imutils.resize(image, width=600)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = image.shape
        step = channel * width
        qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
        imageLabel.setPixmap(QPixmap.fromImage(qImg))
        layout.addWidget(imageLabel)
        self.setLayout(layout)

