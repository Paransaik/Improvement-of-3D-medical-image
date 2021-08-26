import sys
from PyQt5 import QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import pydicom

import numpy as np
import SimpleITK as itk
import qimage2ndarray
import math

import vtk
from Rendering import Rendering
import voxel


class MyWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.lbl_original_img = QGraphicsScene()  # 많은 수의 2D 그래픽 항목을 관리하기 위한 표면을 제공
        self.lbl_blending_img = QGraphicsScene()

        self.view_1 = QGraphicsView(self.lbl_original_img)  # 원본 이미지
        self.view_2 = QGraphicsView(self.lbl_blending_img)  # 변환된 이미지 뷰

        self.lbl_pos = QLabel()  # 비어있는 라벨? 생성 -> mouseMoveEvent()에서 .setText()를 이용해 계속 갱신
        self.lbl_pos.setAlignment(Qt.AlignLeft)  # 라벨을 AlignLeft에 위치시킨다.

        self.draw = QVBoxLayout()  # 오른쪽에 그리는 방법 펜 컬러 버러쉬 컬러 수정하는 창

        gb = QGroupBox('그리기 종류')
        self.draw.addWidget(gb)

        box = QVBoxLayout()
        gb.setLayout(box)

        text = ['Curve', 'Polygon']
        self.radiobtns = []

        for i in range(len(text)):  # Curve, Polygon 위젯 생성
            self.radiobtns.append(QRadioButton(text[i], self))
            self.radiobtns[i].clicked.connect(self.radioClicked)
            box.addWidget(self.radiobtns[i])
        self.radiobtns[0].setChecked(True)
        self.drawType = 0

        gb = QGroupBox('펜 설정')
        self.draw.addWidget(gb)

        grid = QGridLayout()
        gb.setLayout(grid)

        label = QLabel('선굵기')
        grid.addWidget(label, 0, 0)

        self.combo = QComboBox()
        grid.addWidget(self.combo, 0, 1)

        for i in range(1, 21):
            self.combo.addItem(str(i))

        label = QLabel('선색상')
        grid.addWidget(label, 1, 0)

        self.pencolor = QColor(0, 0, 0)
        self.penbtn = QPushButton()
        self.penbtn.setStyleSheet('background-color: rgb(0,0,0)')
        self.penbtn.clicked.connect(self.showColorDlg)
        grid.addWidget(self.penbtn, 1, 1)

        # 그룹박스3
        gb = QGroupBox('붓 설정')
        self.draw.addWidget(gb)

        hbox = QHBoxLayout()
        gb.setLayout(hbox)

        label = QLabel('붓색상')
        hbox.addWidget(label)

        self.brushcolor = QColor(255, 255, 255)
        self.brushbtn = QPushButton()
        self.brushbtn.setStyleSheet('background-color: rgb(255,255,255)')
        self.brushbtn.clicked.connect(self.showColorDlg)
        hbox.addWidget(self.brushbtn)

        # 그룹박스4
        gb = QGroupBox('지우개')
        self.draw.addWidget(gb)

        hbox = QHBoxLayout()
        gb.setLayout(hbox)

        # 밑에 여러가지 기능 버튼 추가
        self.checkbox = QCheckBox('지우개 동작')
        self.checkbox.stateChanged.connect(self.checkClicked)
        hbox.addWidget(self.checkbox)

        self.btn1 = QPushButton('&Undo', self)
        self.btn1.setCheckable(True)  # 버튼을 누른상태와 그렇지 않은 상태를 구분함
        self.btn1.toggle()

        self.btn2 = QPushButton('&SR', self)
        self.btn2.setCheckable(True)
        self.btn2.toggle()

        self.btn3 = QPushButton('&3D Rendering', self)
        self.btn3.setCheckable(True)
        self.btn3.toggle()

        self.btn4 = QPushButton('&Previous', self)
        self.btn4.setCheckable(True)
        self.btn4.toggle()

        self.btn5 = QPushButton('&Next', self)
        self.btn5.setCheckable(True)
        self.btn5.toggle()

        self.btn6 = QPushButton('&ImgNum', self)
        self.btn6.setCheckable(True)
        self.btn6.toggle()

        # 박스 형태를 그림으로 그리기
        self.hbox1 = QHBoxLayout()  # 행( row) 방향(수평)으로 위젯을 배치할 때 사용하는 레이아웃
        self.hbox2 = QHBoxLayout()

        # 행을 기준으로 배치하기 때문에 원본, 변환 이미지 위젯을 추가함
        self.hbox1.addWidget(self.view_1)  # 폼 박스에 원본 이미지 위젯 추가
        self.hbox1.addWidget(self.view_2)  # 폼 박스에 변환 이미지 위젯 추가
        self.hbox1.addLayout(self.draw)

        self.hbox2.addWidget(self.lbl_pos)  # H박스에 위젯 추가 (previous, next, ImgNum 등)
        self.hbox2.addStretch(3)
        self.hbox2.addWidget(self.btn1)
        self.hbox2.addWidget(self.btn2)
        self.hbox2.addWidget(self.btn3)
        self.hbox2.addWidget(self.btn4)
        self.hbox2.addWidget(self.btn5)
        self.hbox2.addWidget(self.btn6)

        self.vbox = QVBoxLayout()  # 열(col) 방향(수직)으로 위젯을 배치할 때 사용
        self.vbox.addLayout(self.hbox1)
        self.vbox.addLayout(self.hbox2)

        # draw레이아웃에 여백 공간 생성
        self.draw.addStretch(1)

        self.setLayout(self.vbox)  # addLayout(self.hbox)를 했기 때문에 setLayout을 vbox로
        # hbox1안에 위젯들 여백 공간 설정
        self.vbox.setStretchFactor(self.hbox1, 1)

    def radioClicked(self):  # 그리는 방법 설정에 뭘로 설정했는지 확인하는 함수
        for i in range(len(self.radiobtns)):
            if self.radiobtns[i].isChecked():
                self.drawType = i
                break

    def checkClicked(self):  # 지우개 부분에 뭘로 설정했는지 확인하는 함수
        pass

    def showColorDlg(self):  # 펜컬러, 브러쉬 컬러 설정하는 함수

        # 색상 대화상자 생성
        color = QColorDialog.getColor()

        sender = self.sender()

        # 색상이 유효한 값이면 참, QFrame에 색 적용
        if sender == self.penbtn and color.isValid():
            self.pencolor = color
            self.penbtn.setStyleSheet('background-color: {}'.format(color.name()))
        else:
            self.brushcolor = color
            self.brushbtn.setStyleSheet('background-color: {}'.format(color.name()))


class AdjustDialog(QDialog):  # 안에 width와 level를 입력받아서 그 값으로 수정하는 Dialog
    def __init__(self):
        super().__init__()
        self.setupUI()

        self.level = int
        self.width = int

    def setupUI(self):
        self.setGeometry(1100, 200, 300, 100)
        self.setWindowTitle("Sign In")
        self.setWindowIcon(QIcon('./icon/Adjust.png'))

        label1 = QLabel("level: ")
        label2 = QLabel("Width: ")

        self.lineEdit1 = QLineEdit()
        self.lineEdit2 = QLineEdit()
        self.pushButton1 = QPushButton("입력 완료")
        self.pushButton1.clicked.connect(self.pushButtonClicked)

        layout = QGridLayout()
        layout.addWidget(label1, 0, 0)
        layout.addWidget(self.lineEdit1, 0, 1)
        layout.addWidget(self.pushButton1, 0, 2)
        layout.addWidget(label2, 1, 0)
        layout.addWidget(self.lineEdit2, 1, 1)

        self.setLayout(layout)

    def pushButtonClicked(self):
        self.level = self.lineEdit1.text()
        self.width = self.lineEdit2.text()
        self.close()

class MyApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.bCtrl = False
        self.zoom = QPointF()

        self.LRpoint = [0, 0]
        self.LRClicked = False
        self.window_level = 2700
        self.window_width = 5350
        self.deltaWL = 0
        self.deltaWW = 0
        # window Center는 보고 싶은 부위의 HU 값을 의미하고, Window Width는 WC
        # Window Center는 -600으로 잡고, Window Width는 1600으로 잡아주면 된다.

        self.Nx = 0  # 이미지의 높이 크기가 저장
        self.Ny = 0  # 이미지의 너비 크기가 저장
        self.NofI = 0  # 총 이미지의 개수가 들어가는 변수 / openImage 메소드에서 한 번 정의된다.

        self.cur_idx = 0  # Pixmap에 올라갈 이미지를 정하는 idx
        self.cur_image = []  # Pixmap에 올라갈 이미지, 왜 리스트 자료형으로 선언했는지 모르겠다.
        self.EntireImage = []  # Pixmap에 올라갈 이미지들을 가지고있는 Image(폴더?), 같은 사진(중복) .dcm파일들의 개수만큼 idx를 가지고 있으며 한 사진마다 blending 등을 수행하고 ImgNum버튼을 통해 각 사진별로 달라진점 확인 가능?
        self.adjustedImage = []  # 현재 사용X
        self.location = []  # polygon의 위치 좌표

        self.mask_space = '1'  # 그림 그리는 페인트 마스크값

        self.vx = voxel.PyVoxel()  # 복셀 생성자 호출

        self.imagePath = ''  # 3D Rendering을 위한 변수 선언
        self.folder_path = ''  # 2D Rendering을 위한 변수 선언

        self.items = []  # 그리는 행동, 폴리곤 만드는 행동을 저장하는 리스트
        self.start = QPointF()  # 그리기 시작한 좌표점
        self.end = QPointF()  # 그리기 끝난 좌표점
        self.polygon = QPoint()  # 폴리곤의 좌표
        self.label = QLabel()  # QLabel 메서드를 들고옴

        self.wg = MyWidget()  # MyWidget 클래스를 사용하기 위해서 객체를 생성
        self.setCentralWidget(self.wg)  # QMainWindow 화면에 레이아웃과 위젯을 표시하기 위해사용
        self.initUI()  # 멤버 메소드 호출

    def initUI(self):
        openDcmAction = QAction(QIcon('./icon/openDcm.png'), 'openDcm', self)
        openRawAndBinAction = QAction(QIcon('./icon/openRawAndBin.png'), 'openRawAndBin', self)
        exportRawAndBinAction = QAction(QIcon('./icon/exportRawAndBin.png'), 'exportRawAndBin', self)
        adjustAction = QAction(QIcon('./icon/Adjust.png'), "Adjust", self)
        # saveAction = QAction(QIcon('./icon/save.png'), 'save', self)
        # optionAction = QAction(QIcon('./icon/option.png'), 'option', self)

        openDcmAction.triggered.connect(self.openDcm)  # 메소드에 매핑
        openRawAndBinAction.triggered.connect(self.openRawAndBin)
        exportRawAndBinAction.triggered.connect(self.exportRawAndBin)
        adjustAction.triggered.connect(self.AdjustDialogClicked)
        # saveAction.triggered.connect(self.saveImage)
        # optionAction.triggered.connect(self.optionImage)

        self.toolbar = self.addToolBar('ToolBar')

        self.toolbar.addAction(openDcmAction)
        self.toolbar.addAction(openRawAndBinAction)
        self.toolbar.addAction(exportRawAndBinAction)
        self.toolbar.addAction(adjustAction)

        self.wg.btn1.setShortcut('Ctrl+z')

        self.wg.btn1.clicked.connect(self.undoButton)  # 버튼 클릭시 연결될 메소드
        self.wg.btn2.clicked.connect(self.s_rButton)
        self.wg.btn3.clicked.connect(self.renderButton)
        self.wg.btn4.clicked.connect(self.previousButton)
        self.wg.btn5.clicked.connect(self.nextButton)
        self.wg.btn6.clicked.connect(self.showDialog)

        self.setWindowTitle('Test Image')  # 타이틀바에 나타나는 창의 제목
        self.setGeometry(100, 100, 1100, 600)  # (move, resize)의 기능을 넣음, resize(w, h) : 위젯의 크기를 너비 w(px), 높이 h(px)로 조절
        self.show()

    # btn1(uodo)가 클릭 되었을 때
    def undoButton(self):
        print(self.items)
        self.items.pop(-1)
        print(self.items)
        self.update()

    # btn2(s_r)가 클릭 되었을 때
    def s_rButton(self):
        return 0

    # btn3(3D Rendering)가 클릭 되었을 때
    def renderButton(self):
        if len(self.EntireImage) == 0:
            print("아직 dataset이 들어오지 않았습니다.")
        else:
            Rendering.VolumeRendering(self.folder_path)  # Volume

    # btn4(Previous)가 클릭 되었을 때
    def previousButton(self):
        if len(self.EntireImage) == 0:
            print("아직 dataset이 들어오지 않았습니다.")
        else:
            self.cur_idx = self.cur_idx - 1  # idx를 1감소
            if self.cur_idx < 0:  # idx가 0보다 작을수는 없다.
                self.cur_idx = 0

            print("left and image", self.cur_idx + 1)
            # EntireImage = (같은 사진 개수, 해상도, 해상도)
            self.cur_image = self.EntireImage[self.cur_idx]  # 해당 idx 번째에 있는 사진을 cur_image로 설정

            image = self.AdjustPixelRange(self.cur_image, self.window_level, self.window_width)
            image = qimage2ndarray.array2qimage(image)
            image = QPixmap.fromImage(QImage(image))

            self.wg.lbl_original_img.addPixmap(image)  # 이전 idx에 있던 이미지를
            self.wg.lbl_blending_img.addPixmap(image)  # pixmap에 올릴 이미지로 변경
            self.wg.view_1.setScene(self.wg.lbl_original_img)
            self.wg.view_2.setScene(self.wg.lbl_original_img)
            self.wg.view_1.show()
            self.wg.view_2.show()

    # btn5(Next)가 클릭 되었을 때
    def nextButton(self):
        if len(self.EntireImage) == 0:
            print("아직 dataset이 들어오지 않았습니다.")
        else:
            self.cur_idx = self.cur_idx + 1  #
            if self.cur_idx > self.NofI - 1:
                self.cur_idx = self.NofI - 1

            print("right and image =", self.cur_idx + 1)
            self.cur_image = self.EntireImage[self.cur_idx]

            image = self.AdjustPixelRange(self.cur_image, self.window_level, self.window_width)
            image = qimage2ndarray.array2qimage(image)
            image = QPixmap.fromImage(QImage(image))

            # 오른쪽 프레임 이미지 업데이트 필요
            self.wg.lbl_original_img.addPixmap(image)
            self.wg.lbl_blending_img.addPixmap(image)
            self.wg.view_1.setScene(self.wg.lbl_original_img)
            self.wg.view_2.setScene(self.wg.lbl_original_img)
            self.wg.view_1.show()
            self.wg.view_2.show()

    # btn6(ImgNum)가 클릭 되었을 때
    # 사용자로부터 단순한 정수를 입력받는 것이 아니라 다양한 옵션 중 하나를 선택하고자 한다면 getItem 메소드를 사용
    def showDialog(self):
        num, ok = QInputDialog.getInt(self, 'Input ImageNumber',
                                      'Enter Num')  # 두 번째 파라미터(타이틀 바 제목), 세 번째 파라미터(EditHint?)
        self.cur_idx = num - 1  # idx로 변환해야하기 때문에 -1
        if len(self.EntireImage) == 0:
            print("아직 dataset이 들어오지 않았습니다.")
        elif self.cur_idx < 0 or self.cur_idx >= len(self.EntireImage):
            print("선택할 수 없는 번호입니다.")
        else:
            print("show image", self.cur_idx + 1)
            if self.cur_idx > self.NofI - 1:  # NofI는 openImage에서 한 번 정의 된다. 총 이미지 개수
                self.cur_idx = self.NofI - 1  # 만약 표시하고 싶은 idx 범위가 사진의 수를 넘어간다면(없는 이미지) 마지막 이미지로 변경
            elif self.cur_idx < 0:  # ?
                self.cur_idx = self.NofI - 224  # 음수 값으로 변경 & 오류 발생 후 종료

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

    def openDcm(self):
        # QFileDialog는 사용자가 파일 또는 경로를 선택할 수 있도록 하는 다이얼로그
        self.imagePath, _ = QFileDialog.getOpenFileName(self, 'Open file',
                                                        './image')  # 'Open file'은 열리는 위젯의 이름, 세 번째 매개변수는 기본 경로설정
        if self.imagePath == '':
            print('openDcm 종료')
        else:
            dcmfileName = self.imagePath.split('/')[-1]  # 현재 보고있는 .dcm파일의 file명
            extendName = dcmfileName[-3:]  # 뒤에 확장자명 조회하기, 확장자 명에 따라 호출되는 함수가 다름
            if extendName == 'dcm' or extendName == 'DCM':  # only open dcm file
                self.folder_path = ''  # 다른 dataset으로의 변경을 위한 초기화
                for i in range(len(self.imagePath.split('/')) - 1):  # folder_path를 imagePath를 이용해서 구해야지만 앞으로 문제 발생 X
                    if i == len(self.imagePath.split('/')) - 1:
                        self.folder_path = self.folder_path + self.imagePath.split('/')[i]
                    else:
                        self.folder_path = self.folder_path + self.imagePath.split('/')[i] + '/'

                reader = itk.ImageSeriesReader()  # reader이름의 객체 생성
                dicom_names = reader.GetGDCMSeriesFileNames(self.folder_path)  # 폴더내에있는 .dcm 파일을 가져온다.

                reader.SetFileNames(dicom_names)
                images = reader.Execute()
                # print('folder_path', self.folder_path)
                # <class 'SimpleITK.SimpleITK.Image'> <class 'SimpleITK.SimpleITK.Image'>
                print(type(images[0]), type(images[1]))

                imgArray = itk.GetArrayFromImage(images)  # 이미지로부터 배열을is_opened  가져옴
                print(imgArray.shape)
                # EntireImage Handler========================================================================
                self.EntireImage = np.asarray(imgArray, dtype=np.float32)  # asarray는 데이터 형태가 다를 경우에만 복사(copy)가 된다.
                self.EntireImage = np.squeeze(
                    self.EntireImage)  # (배열, 축)을 통해 지정된 축의 차원을 축소, (1, 1024, 1024) -> (1024, 1024)

                print(self.EntireImage.shape)

                self.NofI = self.EntireImage.shape[0]  # 같은 이미지 개수
                self.Nx = self.EntireImage.shape[1]
                self.Ny = self.EntireImage.shape[2]

                temp_space = np.zeros(self.EntireImage.shape)
                self.mask_space = self.vx.Create_Mask_Space(temp_space)

                # self.wg.view_1.setFixedSize(self.EntireImage.shape[1],self.EntireImage.shape[2]) # 이미지 크기에 맞게 view를 설정하면 너무 커지는 현상 발생
                # self.wg.view_2.setFixedSize(self.EntireImage.shape[1], self.EntireImage.shape[2])
                print("view size가", self.EntireImage.shape[1], "와", self.EntireImage.shape[2], "로 설정 되었습니다.")

                self.cur_image = self.EntireImage[self.cur_idx]
                # =============================================================================================

                # 보고 싶은 신체 부위가 있다면 HU table을 참고해 Window Center와 Window Width를 조절한 뒤 그 부분 위주로 출력해줄 수 있다.
                # WC를 중심으로 WW의 범위만큼을 중심적으로 표현해준다.
                # window_level이 낮을수록 하얗게 나온다.(HU 조절?), window_width는 WC를 중심으로 관찰하고자 하는 HU 범위를 의미
                image = self.AdjustPixelRange(self.cur_image, self.window_level, self.window_width)
                image = qimage2ndarray.array2qimage(image)  # 배열에서 이미지로
                image = QPixmap.fromImage(
                    QImage(image))  # image를 입력해주고 QPixmap 객체를 하나 만든다. https://wikidocs.net/33768 < 참고하면 좋다.

                self.wg.lbl_original_img.addPixmap(image)  # MyWidget에서 GraphicsScene()로 선언한 변수에 pixmap을 표시될 이미지로 설정
                self.wg.view_1.setScene(self.wg.lbl_original_img)  # MyWidget에서 QGraphicsView()로 선언한 view_1의 화면으로 설정
                self.wg.view_1.show()  # view_1 시작

                self.wg.lbl_blending_img.addPixmap(image)  # 원래는 blending된 image를 넣어야 하지만 아직 blending 기능X
                self.wg.view_2.setScene(self.wg.lbl_blending_img)
                self.wg.view_2.show()

                self.wg.view_1.mouseMoveEvent = self.mouseMoveEvent  # view_1의 mouseMoveEvent 갱신
                self.wg.view_2.mouseMoveEvent = self.mouseMoveEvent
                self.wg.view_1.setMouseTracking(True)  # True일 때는 마우스 이동 감지
                self.wg.view_2.setMouseTracking(True)  # False일 때는 마우스 클릭시에만 이동 감지

    def openRawAndBin(self):
        self.imagePath, _ = QFileDialog.getOpenFileName(self, 'Open file', './raw')  # 'Open file'은 열리는 위젯의 이름, 세 번째 매개변수는 기본 경로설정

        if self.imagePath == '':
            print('openRaw 종료')
        else:
            dcmfileName = self.imagePath.split('/')[-1]  # 현재 보고있는 .dcm파일의 file명
            extendName = dcmfileName[-3:]  # 뒤에 확장자명 조회하기, 확장자 명에 따라 호출되는 함수가 다름
            if extendName == 'raw' or extendName == 'RAW':  # only open raw file
                self.folder_path = ''  # 다른 dataset으로의 변경을 위한 초기화
                for i in range(len(self.imagePath.split('/')) - 1):  # folder_path를 imagePath를 이용해서 구해야지만 앞으로 문제 발생 X
                    if i == len(self.imagePath.split('/')) - 1:
                        self.folder_path = self.folder_path + self.imagePath.split('/')[i]
                    else:
                        self.folder_path = self.folder_path + self.imagePath.split('/')[i] + '/'
                self.vx.ReadFromRaw(self.imagePath)
                imgArray = self.vx.m_Voxel  # 이미지로부터 배열을 가져옴

                # EntireImage Handler========================================================================
                self.EntireImage = np.asarray(imgArray, dtype=np.float32)
                self.EntireImage = np.squeeze(self.EntireImage)

                self.NofI = self.EntireImage.shape[0]
                self.Nx = self.EntireImage.shape[1]
                self.Ny = self.EntireImage.shape[2]

                self.cur_image = self.EntireImage[self.cur_idx]
                # =============================================================================================
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

                try:
                    fileName = self.imagePath.split('/')[-1]  # 현재 보고있는 .dcm파일의 Directory명
                    fileName = fileName[:-4]  # 뒤에 확장자 제거
                    # ex) direName = dataset.raw > dataset

                    # 편의성을 위해(파일 둘다 확인) 임시로 ./raw 로 설정, 나중에 './bin/'으로 바꿀 것 - 태영
                    path = './raw/' + fileName + '.bin'  # path 설정
                    self.vx.ReadFromBin(path)
                except FileNotFoundError:
                    print('현재 해당 .raw에 대한 bin 파일이 존재하지 않습니다.')


    # DCM --> Raw, Bin or Raw, Bin --> Raw, Bin
    # if image is qPixelmap --> numpy array
    def exportRawAndBin(self):
        if self.imagePath == '':
            print('exportRaw 종료')
        else:
            dcmfileName = self.imagePath.split('/')[-1]  # 현재 보고있는 .dcm파일의 file명
            extendName = dcmfileName[-3:]  # 뒤에 확장자명 조회하기, 확장자 명에 따라 호출되는 함수가 다름
            if extendName == 'dcm' or extendName == 'DCM' or extendName == 'raw' or extendName == 'RAW':  # dcm 파일로 열었을 때 raw로 저장하는 곳
                print('if opened {}'.format(extendName))
                direName = self.folder_path.split('/')[-2]  # 현재 보고있는 .dcm파일의 Directory명
                path = './raw/' + direName + '.raw'  # 저장할 path 설정
                self.vx.NumpyArraytoVoxel(self.EntireImage)
                self.vx.WriteToRaw(path)  # dcm으로 연 파일 raw로 저장

                # 편의성을 위해(파일 둘다 확인) 임시로 ./raw 로 설정, 나중에 './bin/'으로 바꿀 것 - 태영
                path = './raw/' + direName + '.bin'  # path 설정
                self.vx.WriteToBin(path)
            # else:                                         # raw 파일로 열었을 때 raw로 저장하는 곳
            #     print('if opened RAW')
            #     direName = self.imagePath.split('/')[-1]  # 현재 보고있는 .dcm파일의 Directory명
            #     direName = direName[:-4]  # 뒤에 확장자 제거
            #     path = './raw/' + direName + '.raw'  # 저장할 path 설정
            #     self.vx.NumpyArraytoVoxel(self.EntireImage)
            #     self.vx.WriteToRaw(path)  # raw로 연 파일 raw로 저장
            #
            #     # 편의성을 위해(파일 둘다 확인) 임시로 ./raw 로 설정, 나중에 './bin/'으로 바꿀 것 - 태영
            #     path = './raw/' + direName + '.bin'  # path 설정
            #     self.vx.WriteToBin(path)

    def AdjustDialogClicked(self):
        dlg = AdjustDialog()
        dlg.exec_()
        self.level = dlg.level
        self.width = dlg.width
        print('1111111', type(self.level))
        # if type(self.level) != float:
        #     print('level, width가 제대로 입력되지 않았습니다.')
        # else:
        self.label.setText("level: %s width: %s" % (self.level, self.width))
        try:
            self.level = int(self.level)
            self.width = int(self.width)

            image = self.AdjustPixelRange(self.cur_image, self.level,
                                          self.width)  # 이미지를 받아서 window의 높이와 너비를 바꿔서 image로 설정

            image = qimage2ndarray.array2qimage(image)  # Q이미지를 numpy array로 바꿈
            image = QPixmap.fromImage(QImage(image))  # numpy array를 pixmap으로 변환
            self.wg.lbl_original_img.addPixmap(image)  # 원본이미지의 Q이미지를 pix맵으로 설정
            self.wg.lbl_blending_img.addPixmap(image)  # 라벨링 되어있는 이미지의 Q이미지를 pix맵으로 설정
            self.wg.view_1.setScene(self.wg.lbl_original_img)  # 원본 이미지를 보도록 설정
            self.wg.view_2.setScene(self.wg.lbl_blending_img)  # 라벨링 할 이미지를 보도록 설정
            self.wg.view_1.show()  # 원본이미지를 띄움
            self.wg.view_2.show()  # 라벨링 이미지를 띄움
        except ValueError:
            print("level, width가 제대로 입력되지 않았습니다.")

    def onChanged(self, text):
        self.lbl.setText(text)  # 마우스 위치를 알려주는 글을 입력
        self.lbl.adjustSize()  # 위에서 설정한 text를 출력

    def AdjustPixelRange(self, image, level, width):  # Hounsfield 조절 함수
        # 수학 식
        Lower = level - (width / 2.0)
        Upper = level + (width / 2.0)

        range_ratio = (Upper - Lower) / 256.0

        img_adjusted = (image - Lower) / range_ratio
        image = img_adjusted.clip(0, 255)  # numpy.clip(array, min, max) min 값 보다 작은 값들을 min으로 바꿔준다.(max도 마찬가지)

        return image

    def mouseMoveEvent(self, event):
        txt = "마우스가 위치한 이미지의 좌표 ; x={0},y={1}".format(event.x(), event.y())
        self.wg.lbl_pos.setText(txt)
        self.wg.lbl_pos.adjustSize()  # 내용에 맞게 위젯의 크기를 조정한다. https://doc.qt.io/qt-5/qwidget.html#adjustSize
        if event.buttons() & QtCore.Qt.LeftButton:  # 그리는 기능
            self.end = event.pos()
            if self.wg.drawType == 0:  # 그리는 방법을 Curve로 설정했을경우 실행
                pen = QPen(QColor(self.wg.pencolor), self.wg.combo.currentIndex())
                line = QLineF(self.start.x(), self.start.y(), self.end.x(), self.end.y())
                self.items.append(self.wg.lbl_blending_img.addLine(line, pen))
                # print(line)
                # 시작점을 다시 기존 끝점으로
                self.start = event.pos()

            if self.wg.drawType == 1:  # 그리는 방법을 polygon으로 설정했을경우 실행
                pen = QPen(QColor(self.wg.pencolor), self.wg.combo.currentIndex())
                brush = QBrush(self.wg.brushcolor)
                polygon = QPolygonF(self.location)
                self.items.append(self.wg.lbl_blending_img.addPolygon(polygon, pen, brush))

        # mousePressEvent에서 클릭을 감지하면 True로 변경
        if self.LRClicked:
            mX = float(event.globalX())  # 이벤트 발생 시 마우스 커서의 전역 x위치를 반환
            mY = float(event.globalY())  # ...

            # LRPoint = [x, y] 초기값은 0, but 클릭을 감지했다는 건 [x, y]는 다른 값으로 이미 갱신
            rX = np.array(self.LRpoint[0])
            rY = np.array(self.LRpoint[1])

            square = (rX - mX) * (rX - mX) + (rY - mY) * (rY - mY)
            dist = math.sqrt(square) * 2  # 거리

            temp_wl = 0
            temp_ww = 0

            if rX < mX:  # X는 WL 값을 변경
                self.deltaWL = dist
            else:
                self.deltaWL = -dist
            if rY < mY:  # Y는 WW 값을 변경
                self.deltaWW = -dist
            else:
                self.deltaWW = dist

            temp_wl = self.window_level + self.deltaWL
            temp_ww = self.window_width + self.deltaWW

            if temp_wl < 0:  # 이 프로젝트에서는 wl과 ww가 음수라면 값을 0으로 맞춰줌
                temp_wl = 0

            if temp_ww < 0:
                temp_ww = 0

            print("move: ", temp_wl, temp_ww)  # 현재 최종 wl와 ww를 출력

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            if self.wg.drawType == 0:
                self.start = event.pos()
                self.start = self.start + QtCore.QPoint(-10, -44)
                self.end = event.pos()

            if self.wg.drawType == 1:  # polygon의 위치를 받고 그것을 self location에 추가함
                self.polygon = event.pos()
                print(self.polygon)
                self.location.append(self.polygon)
                print("좌표가 추가되었습니다")

        if event.buttons() == QtCore.Qt.RightButton:  # 질문 : 이게 왜 동시 클릭을 의미?
            # LRClicked가 False로 되어있다면
            if self.LRClicked == False:
                self.LRClicked = True  # True(클릭)으로 값 바꾸기
            # 이미 LRClicked가 True(클릭)으로 되어있다면, 아마 마우스 버튼 클릭 후 떼는 동작!
            else:
                self.LRClicked = False  # False(no클릭)으로 값 바꾸기

                # wl, ww를 변경시킨다.
                self.window_level = self.window_level + self.deltaWL
                self.window_width = self.window_width + self.deltaWW

                if self.window_level < 0:  # wl과 ww가 음수라면 값을 0으로 맞춰준다.
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
                self.wg.view_2.setScene(self.wg.lbl_blending_img)
                self.wg.view_1.show()
                self.wg.view_2.show()

            print("mousePressEvent")
            print("Mouse 클릭한 글로벌 좌표: x={0},y={1}".format(event.globalX(), event.globalY()))

            x = event.globalX()
            y = event.globalY()

            self.LRpoint = [x, y]

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            if self.wg.checkbox.isChecked():
                return None
            pen = QPen(QColor(self.wg.pencolor), self.wg.combo.currentIndex())

            if self.wg.drawType == 1:
                brush = QBrush(self.wg.brushcolor)

                self.items.clear()
                Polygon = QPolygonF(self.start, self.end)
                self.wg.lbl_blending_img.addPolygon(Polygon, pen, brush)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Control:
            self.bCtrl = True
        self.update()

    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key_Control:
            self.bCtrl = False
        self.update()

    def wheelEvent(self, e):
        if self.bCtrl:
            print(self.zoom.y())  # 이 값에 514 곱하기
            self.zoom += e.angleDelta() / 120
        self.update()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())