import sys
from PyQt5 import QtCore
from PyQt5.QtWidgets import (QApplication, QLabel, QMainWindow, QWidget,
                             QHBoxLayout, QVBoxLayout, QAction, QFileDialog, QGraphicsView, QGraphicsScene)
from PyQt5.QtGui import QPixmap, QIcon, QImage
from PyQt5.QtCore import Qt
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
        super().__init__()  # 부모 클래스(QWidget)의 생성자를 super()를 통해 호출

        # QGraphics를 이용한 ~
        self.lbl_original_img = QGraphicsScene()  # 많은 수의 2D 그래픽 항목을 관리하기 위한 표면을 제공
        self.lbl_blending_img = QGraphicsScene()

        self.view_1 = QGraphicsView(self.lbl_original_img)  # 원본 이미지
        self.view_2 = QGraphicsView(self.lbl_blending_img)  # 변환된 이미지 뷰

        self.view_1.setFixedSize(514, 514)  # 가로, 세로크기를 고정
        self.view_2.setFixedSize(514, 514)  # 다이얼 최소/최대 값의 범위는 setRange()함수를 이용

        self.lbl_pos = QLabel()  # 비어있는 라벨? 생성 -> mouseMoveEvent()에서 .setText()를 이용해 계속 갱신
        self.lbl_pos.setAlignment(Qt.AlignLeft)  # 라벨을 Center에 위치시킨다.

        # 박스 형태를 그림으로 그리기
        self.hbox = QHBoxLayout()  # 행( row) 방향(수평)으로 위젯을 배치할 때 사용하는 레이아웃

        # 행을 기준으로 배치하기 때문에 원본, 변환 이미지 위젯을 추가함
        self.hbox.addWidget(self.view_1)  # 폼 박스에 원본 이미지 위젯 추가
        self.hbox.addWidget(self.view_2)  # 폼 박스에 변환 이미지 위젯 추가

        self.vbox = QVBoxLayout()  # 열(col) 방향(수직)으로 위젯을 배치할 때 사용
        self.vbox.addLayout(self.hbox)
        self.vbox.addWidget(self.lbl_pos)  # V박스에 라벨 위젯 추가 (previous, next, ImgNum)

        self.setLayout(self.vbox)  # addLayout(self.hbox)를 했기 때문에 setLayout을 vbox로

class MyApp(QMainWindow):
    def __init__(self):
        super().__init__()  # QMainWindow의 생성자 호출
        
        self.LRpoint = [0, 0]  # 동시클릭 위치 저장변수
        self.LRClicked = False 
        self.window_level = 2700  # wl
        self.window_width = 5350  # ww
        self.deltaWL = 0  # wl에 더할 delta 값
        self.deltaWW = 0  # ww에 더할 delta 값
        # >? Window Center는 보고 싶은 부위의 HU 값을 의미하고, Window Width는 WC
        # Window Center는 -600으로 잡고, Window Width는 1600으로 잡아주면 된다.

        self.Nx = 0  # 이미지의 높이 크기가 저장, but (1024, 1024)가 (너비, 높이)라면 반대 (3, ㅇ,ㅇ )
        self.Ny = 0  # 이미지의 너비 크기가 저장
        self.NofI = 0  # 총 이미지의 개수가 들어가는 변수 / openImage 메소드에서 한 번 정의된다.

        self.cur_idx = 0  # Pixmap에 올라갈 이미지를 정하는 idx
        self.cur_image = []  # Pixmap에 올라갈 이미지, 왜 리스트 자료형으로 선언했는지 모르겠다.
        self.EntireImage = []  # Pixmap에 올라갈 이미지들을 가지고있는 Image(폴더?), 같은 사진(중복) .dcm파일들의 개수만큼 idx를 가지고 있으며 한 사진마다 blending 등을 수행하고 ImgNum버튼을 통해 각 사진별로 달라진점 확인 가능?
        self.adjustedImage = []  # 현재 사용X

        self.vx = voxel.PyVoxel()  # 복셀 생성자 호출

        self.imagePath = ''  # 3D Rendering을 위한 변수 선언
        self.folder_path = ''  # 2D Rendering을 위한 변수 선언

        self.wg = MyWidget()  # MyWidget 클래스를 사용하기 위해서 객체를 생성
        self.setCentralWidget(self.wg)  # QMainWindow 화면에 레이아웃과 위젯을 표시하기 위해사용
        self.initUI()  # 멤버 메소드 호출
        
    def initUI(self):
        openAction = QAction(QIcon('./icon/open.png'), 'open', self)
        # saveAction = QAction(QIcon('./icon/save.png'), 'save', self)
        # optionAction = QAction(QIcon('./icon/option.png'), 'option', self)

        openAction.triggered.connect(self.openImage)
        # saveAction.triggered.connect(self.saveImage)
        # optionAction.triggered.connect(self.optionImage)

        self.toolbar = self.addToolBar('ToolBar')

        self.toolbar.addAction(openAction)
        # self.toolbar.addAction(saveAction)
        # self.toolbar.addAction(optionAction)

        # 푸시 버튼 또는 명렁 버튼은 사용자가 프로그램에 명령을 내려서 어떤 동작을 하도록 할 때 사용되는 버튼
        btn1 = QPushButton('&Stop', self)  # &은 단축키를 지정하기 위해서 첫 번째 파라미터는 버튼에 나타날 텍스트, 두 번째 파라미터는 버튼이 속할 부모 클래스
        btn1.move(400, 565)  # 위젯을 스크린의 x=900px, y=300px의 위치로 이동
        btn1.setCheckable(True)  # 누른 상태와 그렇지 않은 상태를 구분한다.
        btn1.toggle()  # 버튼의 상태가 바뀌게 된다. 따라서 이 버튼은 프로그램이 시작될 때 선택되어 있다.

        btn2 = QPushButton('&S-R', self)
        btn2.move(500, 565)
        btn2.setCheckable(True)
        btn2.toggle()

        btn3 = QPushButton('&3D Rendering', self)
        btn3.move(600, 565)
        btn3.setCheckable(True)
        btn3.toggle()

        btn4 = QPushButton('&Previous', self)
        btn4.move(700, 565)
        btn4.setCheckable(True)
        btn4.toggle()

        btn5 = QPushButton('&Next', self)
        btn5.move(800, 565)
        btn5.setCheckable(True)
        btn5.toggle()

        btn6 = QPushButton('&ImgNum', self)
        btn6.move(900, 565)
        btn6.setCheckable(True)
        btn6.toggle()

        # btn1.setShortcut('Ctrl+1')
        # btn2.setShortcut('Ctrl+2')

        btn1.clicked.connect(self.stopButton) # 버튼 클릭시 연결될 메소드
        btn2.clicked.connect(self.s_rButton)
        btn3.clicked.connect(self.renderButton)
        btn4.clicked.connect(self.previousButton)
        btn5.clicked.connect(self.nextButton)
        btn6.clicked.connect(self.showDialog)

        self.setWindowTitle('Test Image') # 타이틀바에 나타나는 창의 제목
        self.setGeometry(100, 100, 1100, 600) # (move, resize)의 기능을 넣음, resize(w, h) : 위젯의 크기를 너비 w(px), 높이 h(px)로 조절
        self.show()

    # btn1(stop)가 클릭 되었을 때
    def stopButton(self):
        return 0

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
    # (Entireimage에는 같은 .dcm 파일의 개수만큼 idx가 있고, btn1과 btn2를 누르며 사진을 옮겨가며 확인 가능하다(현재는 이미지에 변화를 주지 않았기 때문에 다 같은 장면으로 보이는 것뿐))
    def previousButton(self):
        if len(self.EntireImage) == 0:
            print("아직 dataset이 들어오지 않았습니다.")
        else:
            self.cur_idx = self.cur_idx - 1  # idx를 1감소
            if self.cur_idx < 0:  # idx가 0보다 작을수는 없다.
                self.cur_idx = 0

            print("left and image", self.cur_idx + 1)  # 이전 image는 몇 번쨰 idx이라고 표시
            # EntireImage = (같은 사진 개수, 해상도, 해상도)
            self.cur_image = self.EntireImage[self.cur_idx]  # 해당 idx 번째에 있는 사진을 cur_image로 설정

            image = self.AdjustPixelRange(self.cur_image, self.window_level, self.window_width)
            image = qimage2ndarray.array2qimage(image)
            image = QPixmap.fromImage(QImage(image))

            self.wg.lbl_original_img.addPixmap(image)  # 이전 idx에 있던 이미지를
            self.wg.lbl_blending_img.addPixmap(image)  # pixmap에 올릴 이미지로 변경
            self.wg.view_1.setScene(self.wg.lbl_original_img)
            self.wg.view_2.setScene(self.wg.lbl_original_img)  # wh.lbl_blending_img는 어디갔찌
            self.wg.view_1.show()
            self.wg.view_2.show()

    # btn5(Next)가 클릭 되었을 때
    def nextButton(self):
        if len(self.EntireImage) == 0:
            print("아직 dataset이 들어오지 않았습니다.")
        else:
            self.cur_idx = self.cur_idx + 1 # 다음 이미지 idx
            if self.cur_idx > self.NofI-1:
                self.cur_idx = self.NofI-1

            print("right and image=", self.cur_idx +1)
            self.cur_image = self.EntireImage[self.cur_idx]

            image = self.AdjustPixelRange(self.cur_image, self.window_level, self.window_width)
            image = qimage2ndarray.array2qimage(image)
            image = QPixmap.fromImage(QImage(image))

            #왼쪽 프레임 이미지 업데이트 필요
            self.wg.lbl_original_img.addPixmap(image)
            self.wg.lbl_blending_img.addPixmap(image)
            self.wg.view_1.setScene(self.wg.lbl_original_img)
            self.wg.view_2.setScene(self.wg.lbl_original_img) # wg.lbl_blending_img는 어디갔나
            self.wg.view_1.show()
            self.wg.view_2.show()

    # btn6(ImgNum)가 클릭 되었을 때
    # 사용자로부터 단순한 정수를 입력받는 것이 아니라 다양한 옵션 중 하나를 선택하고자 한다면 getItem 메소드를 사용
    def showDialog(self):
        num, ok = QInputDialog.getInt(self, 'Input ImageNumber', 'Enter Num')  # 두 번째 파라미터(타이틀 바 제목), 세 번째 파라미터(EditHint?)
        self.cur_idx = num - 1  # idx로 변환해야하기 때문에 -1

        if len(self.EntireImage) == 0:
            print("아직 dataset이 들어오지 않았습니다.")
        elif self.cur_idx < 0 or self.cur_idx >= len(self.EntireImage):
            print("선택할 수 없는 번호입니다.")
        else:
            print("show image", self.cur_idx + 1)  # 해당 idx의 이미지를 보여준다.
            if self.cur_idx > self.NofI - 1:  # NofI는 openImage에서 한 번 정의 된다. Entireimage[0] : 총 이미지 개수?
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

    # 이미지 불러올 때 itk사용
    def openImage(self):
        # QFileDialog는 사용자가 파일 또는 경로를 선택할 수 있도록 하는 다이얼로그
        self.imagePath, _ = QFileDialog.getOpenFileName(self, 'Open file', './image')  # 'Open file'은 열리는 위젯의 이름, 세 번째 매개변수는 기본 경로설정

        if self.imagePath == '':
            print('openImage 종료')
        else:
            self.folder_path = ''  # 다른 dataset으로의 변경을 위한 초기화
            for i in range(len(self.imagePath.split('/')) - 1):  # folder_path를 imagePath를 이용해서 구해야지만 앞으로 문제 발생 X
                if i == len(self.imagePath.split('/')) - 1:
                    self.folder_path = self.folder_path + self.imagePath.split('/')[i]
                else:
                    self.folder_path = self.folder_path + self.imagePath.split('/')[i] + '/'

            reader = itk.ImageSeriesReader()  # reader이름의 객체 생성
            dicom_names = reader.GetGDCMSeriesFileNames(self.folder_path)  # 폴더내에있는 .dcm 파일을 가져온다.

            reader.SetFileNames(dicom_names)
            images = reader.Execute()  # reader의 FileNames를 실행?
            print('folder_path', self.folder_path)
            # <class 'SimpleITK.SimpleITK.Image'> <class 'SimpleITK.SimpleITK.Image'>
            print(type(images[0]), type(images[1]))

            ImgArray = itk.GetArrayFromImage(images)  # 이미지로부터 배열을 가져옴
            print(ImgArray.shape)
            self.EntireImage = np.asarray(ImgArray, dtype=np.float32)  # asarray는 데이터 형태가 다를 경우에만 복사(copy)가 된다.

            self.EntireImage = np.squeeze(
                self.EntireImage)  # (배열, 축)을 통해 지정된 축의 차원을 축소, (1, 1024, 1024) -> (1024, 1024)

            print(self.EntireImage.shape)  # (6, 1024, 1024)

            # 같은 이미지의 .dcm이 두 개가 된다면 EntireImage가 (2, n, n)이되서 프로그램 오류x
            # 또한 같은 이미지의 .dcm 쌍이 두 개가 존재한다면 나머지 한 개의 이미지는 출력되지 않음..
            self.NofI = self.EntireImage.shape[0]  # 같은 이미지 개수
            self.Nx = self.EntireImage.shape[1]  # Nx에서 받아주기는 하지만 [1]은 높이(y)가 아닌가?
            self.Ny = self.EntireImage.shape[2]  # Ny에서 받아주기는 하지만 [2]는 너비(x)가 아닌가?

            self.cur_image = self.EntireImage[self.cur_idx]  # cur_image는 pixmap에 올라갈 image, cur_idx는 EntireImage에서 몇 번째 이미지를 올릴지 정하는 idx

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
            self.wg.view_2.mouseMoveEvent = self.mouseMoveEvent  # ...
            self.wg.view_1.setMouseTracking(True)  # True일 때는 마우스 이동 감지
            self.wg.view_2.setMouseTracking(True)  # False일 때는 마우스 클릭시에만 이동 감지

            # 임시
            # fileName = self.folder_path.split('/')[-2]  # 현재 보고있는 .dcm파일의 Directory명
            # path = './raw/' + fileName + '.raw'  # path 설정
            # self.export_raw(path)

    def export_raw(self, path):  # export_raw로 connect된 버튼 하나 만들기
        # if image is qPixelmap --> numpy array
        print(type(self.EntireImage))  # 해당 dataset의 .dcm의 array형태가 모여있는 변수
        print(self.EntireImage.shape)  # dataset1 기준 (20,512,512)
        self.pyvoxel.NumpyArraytoVoxel(self.EntireImage)
        self.pyvoxel.WriteToRaw(path)

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

            print("move: ", temp_wl, temp_ww) # 현재 최종 wl와 ww를 출력

    # CT 이미지의 width 변경?! 제대로 된 확인이 안된다면 test.py에서 구현해보기
    def mousePressEvent(self, event):
        # event.buttons() = 이벤트가 생성되었을 때 버튼 상태를 반환
        # LeftButton, RightButton, MiddleButton의 조합으로 자주 사용 OR 연산자를 사용
        # 만약 마우스의 왼쪽 오른쪽 동시에 눌렀다면 ~
        # print("t", QtCore.Qt.LeftButton) =  1 출력됌
        if event.buttons() == QtCore.Qt.LeftButton | QtCore.Qt.RightButton:  # 질문 : 이게 왜 동시 클릭을 의미?
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

            self.LRpoint = [x, y]  # 동시에 클릭했다면 x, y 갱신

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())