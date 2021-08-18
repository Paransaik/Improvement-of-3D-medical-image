import sys
from PyQt5 import QtCore
from PyQt5.QtWidgets import (QApplication, QLabel, QMainWindow, QWidget, QHBoxLayout,
                             QVBoxLayout, QAction, QFileDialog, QGraphicsView, QGraphicsScene)
from PyQt5.QtGui import QPixmap, QIcon, QImage
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
import pydicom

import numpy as np
import SimpleITK as itk
import qimage2ndarray
import math


# asdfasdfasdfsdf
class MyWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.lbl_original_img = QGraphicsScene()
        self.lbl_blending_img = QGraphicsScene()
        self.view_1 = QGraphicsView(self.lbl_original_img)
        self.view_2 = QGraphicsView(self.lbl_blending_img)
        self.view_1.setFixedSize(514, 514)
        self.view_2.setFixedSize(514, 514)

        self.lbl_pos = QLabel()
        self.lbl_pos.setAlignment(Qt.AlignCenter)

        self.hbox = QHBoxLayout()

        self.hbox.addWidget(self.view_1)
        self.hbox.addWidget(self.view_2)

        self.vbox = QVBoxLayout()
        self.vbox.addLayout(self.hbox)
        self.vbox.addWidget(self.lbl_pos)

        self.setLayout(self.vbox)


class MyApp(QMainWindow):

    def __init__(self):
        super().__init__()

        self.LRpoint = [0, 0]
        self.LRClicked = False
        self.window_level = 40
        self.window_width = 400
        self.deltaWL = 0
        self.deltaWW = 0

        self.Nx = 0
        self.Ny = 0
        self.NofI = 0

        self.cur_idx = 0
        self.cur_image = []
        self.EntireImage = []
        self.adjustedImage = []

        self.wg = MyWidget()
        self.setCentralWidget(self.wg)
        self.initUI()

    def initUI(self):

        openAction = QAction(QIcon('exit.png'), 'Open', self)
        openAction.triggered.connect(self.openImage)
        self.toolbar = self.addToolBar('Open')
        self.toolbar.addAction(openAction)

        Dbtn = QPushButton('&ImgNum', self)
        Dbtn.move(900, 565)
        Dbtn.setCheckable(True)
        Dbtn.toggle()
        Dbtn.clicked.connect(self.showDialog)

        btn1 = QPushButton('&previous', self)
        btn1.move(700, 565)
        btn1.setCheckable(True)
        btn1.toggle()

        btn2 = QPushButton('&next', self)
        btn2.move(800, 565)
        btn1.setCheckable(True)
        btn1.toggle()

        btn1.setShortcut('Ctrl+1')
        btn2.setShortcut('Ctrl+2')
        btn1.clicked.connect(self.btn1_clicked)
        btn2.clicked.connect(self.btn2_clicked)

        self.setWindowTitle('Test Image')

        self.setGeometry(300, 300, 1100, 600)

        self.show()

    def showDialog(self):
        num, ok = QInputDialog.getInt(self, 'Input ImageNumber', 'Enter Num')
        self.cur_idx = num - 1

        print("show image", self.cur_idx + 1)
        if self.cur_idx > self.NofI - 1:
            self.cur_idx = self.NofI - 1
        elif self.cur_idx < 0:
            self.cur_idx = self.NofI - 224

        self.cur_image = self.EntireImage[self.cur_idx]

        image = self.AdjustPixelRange(self.cur_image, self.window_level, self.window_width)

        image = qimage2ndarray.array2qimage(image)
        image = QPixmap.fromImage(QImage(image))
        self.wg.lbl_original_img.addPixmap(image)
        self.wg.lbl_blending_img.addPixmap(image)
        self.wg.view_1.setScene(self.wg.lbl_original_img)
        self.wg.view_2.setScene(self.wg.lbl_blending_img)
        self.wg.view_1.show()
        self.wg.view_2.show()

    def onChanged(self, text):
        self.lbl.setText(text)
        self.lbl.adjustSize()

    def btn1_clicked(self):

        self.cur_idx = self.cur_idx - 1

        if self.cur_idx < 0:
            self.cur_idx = 0

        print("left and image", self.cur_idx + 1)

        self.cur_image = self.EntireImage[self.cur_idx]

        image = self.AdjustPixelRange(self.cur_image, self.window_level, self.window_width)

        image = qimage2ndarray.array2qimage(image)
        image = QPixmap.fromImage(QImage(image))

        self.wg.lbl_original_img.addPixmap(image)
        self.wg.lbl_blending_img.addPixmap(image)
        self.wg.view_1.setScene(self.wg.lbl_original_img)
        self.wg.view_2.setScene(self.wg.lbl_original_img)
        self.wg.view_1.show()
        self.wg.view_2.show()

    def btn2_clicked(self):

        self.cur_idx = self.cur_idx + 1

        if self.cur_idx > self.NofI - 1:
            self.cur_idx = self.NofI - 1

        print("right and image=", self.cur_idx + 1)

        self.cur_image = self.EntireImage[self.cur_idx]

        image = self.AdjustPixelRange(self.cur_image, self.window_level, self.window_width)

        image = qimage2ndarray.array2qimage(image)
        image = QPixmap.fromImage(QImage(image))

        # 왼쪽 프레임 이미지 업데이트 필요
        self.wg.lbl_original_img.addPixmap(image)
        self.wg.lbl_blending_img.addPixmap(image)
        self.wg.view_1.setScene(self.wg.lbl_original_img)
        self.wg.view_2.setScene(self.wg.lbl_original_img)
        self.wg.view_1.show()
        self.wg.view_2.show()

    def AdjustPixelRange(self, image, level, width):
        Lower = level - (width / 2.0)
        Upper = level + (width / 2.0)

        range_ratio = (Upper - Lower) / 256.0

        img_adjusted = (image - Lower) / range_ratio
        image = img_adjusted.clip(0, 255)

        return image

    def openImage(self):
        imagePath, _ = QFileDialog.getOpenFileName(self, 'Open file', './')

        print("open", imagePath)

        folder_path = "E:\Project\label"

        reader = itk.ImageSeriesReader()

        dicom_names = reader.GetGDCMSeriesFileNames(folder_path)

        print(type(dicom_names))
        reader.SetFileNames(dicom_names)

        images = reader.Execute()

        print(type(images[0]), type(images[1]))

        ImgArray = itk.GetArrayFromImage(images)
        self.EntireImage = np.asarray(ImgArray, dtype=np.float32)
        self.EntireImage = np.squeeze(self.EntireImage)

        print(self.EntireImage.shape)

        self.NofI = self.EntireImage.shape[0]
        self.Nx = self.EntireImage.shape[1]
        self.Ny = self.EntireImage.shape[2]

        self.cur_image = self.EntireImage[self.cur_idx]

        image = self.AdjustPixelRange(self.cur_image, self.window_level, self.window_width)

        image = qimage2ndarray.array2qimage(image)
        image = QPixmap.fromImage(QImage(image))

        self.wg.lbl_original_img.addPixmap(image)
        self.wg.view_1.setScene(self.wg.lbl_original_img)
        self.wg.view_1.show()

        self.wg.lbl_blending_img.addPixmap(image)
        self.wg.view_2.setScene(self.wg.lbl_blending_img)
        self.wg.view_2.show()

        self.wg.view_1.mouseMoveEvent = self.mouseMoveEvent
        self.wg.view_2.mouseMoveEvent = self.mouseMoveEvent
        self.wg.view_1.setMouseTracking(True)
        self.wg.view_2.setMouseTracking(True)

    def mouseMoveEvent(self, event):
        txt = "마우스가 위치한 이미지의 좌표 ; x={0},y={1}".format(event.x(), event.y())
        self.wg.lbl_pos.setText(txt)
        self.wg.lbl_pos.adjustSize()

        if self.LRClicked:

            mX = float(event.globalX())
            mY = float(event.globalY())

            rX = np.array(self.LRpoint[0])
            rY = np.array(self.LRpoint[1])

            square = (rX - mX) * (rX - mX) + (rY - mY) * (rY - mY)
            dist = math.sqrt(square)

            temp_wl = 0
            temp_ww = 0

            if rX < mX:
                self.deltaWL = dist

            else:
                self.deltaWL = -dist

            if rY < mY:
                self.deltaWW = -dist

            else:
                self.deltaWW = dist

            temp_wl = self.window_level + self.deltaWL
            temp_ww = self.window_width + self.deltaWW

            if temp_wl < 0:
                temp_wl = 0

            if temp_ww < 0:
                temp_ww = 0

            print("move: ", temp_wl, temp_ww)

    def mousePressEvent(self, event):

        if event.buttons() == QtCore.Qt.LeftButton | QtCore.Qt.RightButton:

            if self.LRClicked == False:
                self.LRClicked = True

            else:
                self.LRClicked = False

                self.window_level = self.window_level + self.deltaWL
                self.window_width = self.window_width + self.deltaWW

                if self.window_level < 0:
                    self.window_level = 0
                if self.window_width < 0:
                    self.window_width = 0

                print("최종반영 ", self.window_level, self.window_width)

                image = self.AdjustPixelRange(self.cur_image, self.window_level, self.window_width)

                image = qimage2ndarray.array2qimage(image)
                image = QPixmap.fromImage(QImage(image))

                self.wg.lbl_original_img.addPixmap(image)
                self.wg.lbl_blending_img.addPixmap(image)
                self.wg.view_1.setScene(self.wg.lbl_original_img)
                self.wg.view_2.setScene(self.wg.lbl_original_img)
                self.wg.view_1.show()
                self.wg.view_2.show()

            print("mousePressEvent")
            print("Mouse 클릭한 글로벌 좌표: x={0},y={1}".format(event.globalX(), event.globalY()))

            x = event.globalX()
            y = event.globalY()

            self.LRpoint = [x, y]


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())
