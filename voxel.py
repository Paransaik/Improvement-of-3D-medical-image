import numpy as np
import sys

# b는 바이너리의 약자 
class PyVoxel: # Voxel은 Volumn과 pixel의 합성어
    # Raw 이미지 파일은 이미지 파일 포맷 중 하나로 디지털 카메라나 이미지 스캐너의 이미지 센서로부터 최소한으로 처리한 데이터를 포함하고있다.
    # 전혀 가공되지 않은 상태이며 촬상매체에 감지된 빛의 세기에 대한 정보만을 담고있다.
    # raw는 JPEG보다 수많은 이점을 가지고 있다.
    # 더 높은 화질, 무손실 압축, 섬세한 제어, 색 공간은 사용자가 원하는 대로 설정할 수 있다.

    # BIN파일은 다른 회사의 디스크 이미지 개발 및 편집 응용 프로그램의 다양한 만든 파일이다. 
    # 이 파일의 내용은 바이너리 형태이다. 
    # 텍스트 파일이아닌 컴퓨터 파일이다. 
    # 바이너리파일은 '한 줄에 해당하는 데이터를 읽어라'와 같은 명령을 사용하지 않고 '10 바이트를 읽어라'로 사용
    def __init__(self):
        self.m_Org = -1  # 지금 신경안써도된다. 다 255로 되어있을 것이다. , 순서가 뒤바꼈을 경우를 대비한것이다.
        self.m_nX = 0  # 픽셀 개수
        self.m_nY = 0  # ...
        self.m_nZ = 0  #... # 슬라이스 개수

        self.m_fXSp = 1.0  # Spacing(sp) : 사진끼리 떨어져있는 거리(x) # row 파일 픽셀 개수
        self.m_fYSp = 1.0  # ...(y)
        self.m_fZSp = 1.0  # ...(z)

        self.m_fXOrg = -1.0  # Org면 오리지널?
        self.m_fYOrg = -1.0
        self.m_fZOrg = -1.0
        self.m_Voxel = []

    def initialize(self):  # 변수들의 초기 내용을 설정
        self.m_Org = -1
        self.m_nX = 0
        self.m_nY = 0
        self.m_nZ = 0

        self.m_fXSp = 1.0  # Spacing(sp) : 사진끼리 떨어져있는 거리(x)
        self.m_fYSp = 1.0  # ...(y)
        self.m_fZSp = 1.0  # ...(z)

        self.m_fXOrg = -1.0
        self.m_fYOrg = -1.0
        self.m_fZOrg = -1.0
        self.m_Voxel = []

    def ReadFromRaw(self, filename):  # 이미지 16비트
        with open(filename, 'rb') as f:  # rb = byte 형식으로 파일 읽기 # f.close를 매번하기 귀찮기 때문에 with를 사용, f는 instance
            try:
                Header = np.tofile(f, dtype='int32', count=1)  # dtype이 int32인 data만 file로부터 받아오는?
                self.m_Org = Header[0]  # header는 Raw파일에 첫 두바이트를 m_org를 넣었다. 저게 -1이라면
                                        # 만약 -1이 아니라면 숫자란 이야기다 256이 들어있다.
                if self.m_Org == -1:  # 데이터가 거꾸로 뒤집어진? 경우
                    Header = np.fromfile(f, dtype='float32', count=6)  # "x, y, z spacing", "x, y, z orgin"

                    self.m_fXSp = Header[0]
                    self.m_fYSp = Header[1]
                    self.m_fZSp = Header[2]

                    self.m_fXOrg = Header[3]
                    self.m_fYOrg = Header[4]
                    self.m_fZOrg = Header[5]

                    Header = np.fromfile(f, dtype='int32', count=1)  # "nX"
                    self.m_nX = Header[0]
                else:  # 정상일 경우
                    self.m_nX = self.m_Org  # m_nX는 X크기 but, Header[1] = m_nX 값일텐데 왜?

                Header = np.fromfile(f, dtype='int32', count=2)  # nY nZ
                self.m_nY = Header[0]
                self.m_nZ = Header[1]

                Data = np.fromfile(f, dtype='int16', count=self.m_nX*self.m_nY*self.m_nZ)
                self.m_Voxel = np.reshape(Data, (self.m_nZ, self.m_nY, self.m_nX))

            except IOError:
                print('Could not read file ' + filename)
                sys.exit()

    def ReadFromBin(self, filename): # .bin 파일 읽는
            with open(filename, 'rb') as f:
                # print('success ReadFromBin')
                try:
                    Header = np.fromfile(f,dtype='int32', count=1)
                    self.m_Org = Header[0]

                    if self.m_Org == -1:
                        Header = np.fromfile(f, dtype='float32', count=6)
                        self.m_fXSp = Header[0]
                        self.m_fYSp = Header[1]
                        self.m_fZSp = Header[2]

                        self.m_fXOrg = Header[3]
                        self.m_fYOrg = Header[4]
                        self.m_fZOrg = Header[5]

                        Header = np.fromfile(f, dtype='int32', count=1)
                        self.m_nX = Header[0]
                    else:
                        self.m_nX = self.m_Org

                    Header = np.fromfile(f, dtype='int32', count=2)
                    self.m_nY = Header[0]
                    self.m_nZ = Header[1]

                    Data = np.fromfile(f, dtype='uint8', count=self.m_nX*self.m_nY*self.m_nZ)
                    self.m_Voxel = np.reshape(Data, (self.m_nZ, self.m_nY, self.m_nX))

                except IOError:
                    print('Could not read file' + filename)
                    sys.exit()

    def WriteToRaw(self, filename):
        # Header에는 영상의 크기, 컬러의 수, 펠리트 등 다양한 정보들이 들어있다.(기본적으로 가로, 세로 크기와 color정보)
        # 이미지를 출력할 때 반드시 필요한 정보가있다. 그래서 사용자의 정보 입력이 없이는 raw파일을 출력할 수 없다.
        HeaderDim = np.array([self.m_Org, self.m_nX, self.m_nY, self.m_nZ], dtype=np.int32)  # shape = (4,)
        HeaderSpOrg = np.array([self.m_fXSp, self.m_fYSp, self.m_fZSp, self.m_fXOrg, self.m_fYOrg, self.m_fZOrg], dtype=np.float32)  # shape = (6,)

        Save = self.m_Voxel.astype(np.int16, copy=False)  # if the type of m_voxel is np.int16, copy operation isn't perform

        with open(filename, 'wb') as f:  # 쓰기 전용으로 파일 오픈
            HeaderDim[0].tofile(f)  # self.m_Org를 넣는다.
            HeaderSpOrg.tofile(f)  # HeaderSpOrg
            HeaderDim[1:].tofile(f)  # HeaderDim의 값을 넣는다.
            Save.tofile(f)  # .
            
    def WriteToBin(self, filename):
        print('success WriteToBin')
        HeaderDim = np.array([self.m_Org, self.m_nX, self.m_nY, self.m_nZ], dtype=np.int32)
        HeaderSpOrg = np.array([self.m_fXSp, self.m_fYSp, self.m_fZSp, self.m_fXOrg, self.m_fYOrg, self.m_fZOrg], dtype=np.float32)

        Save = self.m_Voxel.astype(np.uint8, copy=False)

        with open(filename, 'wb') as f:
            HeaderDim[0].tofile(f)
            HeaderSpOrg.tofile(f)
            HeaderDim[1:].tofile(f)
            Save.tofile(f)

    def AdjustPixelRange(self, Lower, Upper):  # WL, WW를 설정
        # version 2 
        range_ratio = (Upper - Lower) / 256.0
        pData = self.m_Voxel

        img_adjusted = (pData - Lower)/range_ratio
        img_adjusted = img_adjusted.clip(0, 255)
        return img_adjusted

    def NumpyArraytoVoxel(self, data):  # 넘파이 배열을 받으면
        self.initialize()

        Dim = data.shape  # shape = (,)
        self.m_nX = Dim[2]  # shape는 (가로, 세로)가 아닌 (세로, 가로)의 순서이기 때문
        self.m_nY = Dim[1]
        self.m_nZ = Dim[0] 
        
        self.m_Voxel = data.astype(np.int16, copy=-False)  # .append로하면 겉에는 list가 된다.

    def ConvertValue(self, SrcV, TarV): # 값을 변환시키는 함수? 
        idx = self.m_Voxel == SrcV
        self.m_Voxel[idx] = TarV

    def SaveWithoutHeader(self, filename):  # Header 없이 저장하는 버전
        print('success SaveWithoutHeader')
        Save = self.m_Voxel.astype(np.uint8, copy=False)
        print(Save)
        with open(filename, 'wb') as f:
            Save.tofile(f)

    def Normalize(self):
        self.m_Voxel = self.m_Voxel.astype(np.float32, copy=False)
        maxvalue = np.max(self.m_Voxel)  # m_Voxel의 가장 큰 값을 뽑아낸다.
        print(maxvalue)
        self.m_Voxel = self.m_Voxel/maxvalue  # 0.~으로 정규화 된다.

    def NormalizeMM(self):
        self.m_Voxel = self.m_Voxel.astype(np.float32, copy=False)
        maxvalue = np.max(self.m_Voxel)  # m_Voxel의 가장 큰 값을 뽑아낸다.
        minvalue = np.min(self.m_Voxel)  # m_Voxel의 가장 작은 값을 뽑아낸다.

        diff = maxvalue - minvalue
        self.m_Voxel = (self.m_Voxel - minvalue)/diff

    def AdjustPixelRangeNormalize(self, Upper):  # version 1?
        self.m_Voxel = self.m_Voxel.astype(np.float32, copy=False)
        minvalue = np.min(self.m_Voxel)
        self.m_Voxel = self.m_Voxel.clip(minvalue, Upper)

        diff = Upper - minvalue
        self.m_Voxel = (self.m_Voxel - minvalue)/diff