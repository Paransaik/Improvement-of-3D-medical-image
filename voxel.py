import numpy as np
import sys
# test
class PyVoxel:
    def __init__(self):
        self.m_Org = -1
        self.m_nX = 0
        self.m_nY = 0
        self.m_nZ = 0 # 슬라이스 개수

        self.m_fXSp = 1.0 # row 파일 픽셀 개수
        self.m_fYSp = 1.0
        self.m_fZSp = 1.0

        self.m_fXOrg = -1.0
        self.m_fYOrg = -1.0
        self.m_fZOrg = -1.0
        self.m_Voxel = []

    def initialize(self):
        self.m_Org = -1
        self.m_nX = 0
        self.m_nY = 0
        self.m_nZ = 0

        self.m_fXSp = 1.0
        self.m_fYSp = 1.0
        self.m_fZSp = 1.0

        self.m_fXOrg = -1.0
        self.m_fYOrg = -1.0
        self.m_fZOrg = -1.0
        self.m_Voxel = []

    def ReadFromRaw(self, filename):
        with open(filename, 'rb') as f: # rb = byte 형식으로 파일 읽기
            try:
                Header = np.tofile(f, dtype='int32', count=1)
                self.m_Org = Header[0]

                if self.m_Org == -1:
                    Header = np.fromfile(f, dtype='float32', count=6) # "x, y, z spacing", "x, y, z orgin"
                    self.m_fXSp = Header[0]
                    self.m_fYSp = Header[1]
                    self.m_fZSp = Header[2]

                    self.m_fXOrg = Header[3]
                    self.m_fYOrg = Header[4]
                    self.m_fZOrg = Header[5]

                    Header = np.fromfile(f, dtype='int32', count=1) # "nX"
                    self.m_nX = Header[0]
                else:
                    self.m_nX = self.m_Org

                Header = np.fromfile(f, dtype='int32', count=2) # nY nZ
                self.m_nY = Header[0]
                self.m_nZ = Header[1]

                Data = np.fromfile(f, dtype='int16', count=self.m_nX*self.m_nY*self.m_nZ)
                self.m_Voxel = np.reshape(Data, (self.m_nZ, self.m_nY, self.m_nX))

            except IOError:
                print('Could not read file ' + filename)
                sys.exit()

    def ReadFromBin(self, filename):
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
        print('success WriteToRaw')
        print(filename)
        HeaderDim = np.array([self.m_Org, self.m_nX, self.m_nY, self.m_nZ], dtype=np.int32)
        HeaderSpOrg = np.array([self.m_fXSp, self.m_fYSp, self.m_fZSp, self.m_fXOrg, self.m_fYOrg, self.m_fZOrg], dtype=np.float32)
        print(HeaderDim)
        print(HeaderSpOrg)

        Save = self.m_Voxel.astype(np.int16, copy=False) # if the type of m_voxel is np.int16, copy operation isn't perform

        with open(filename, 'wb') as f:
            HeaderDim[0].tofile(f)
            HeaderSpOrg.tofile(f)
            HeaderDim[1:].tofile(f)
            Save.tofile(f)

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

    def AdjustPixelRange(self, Lower, Upper):
        # version 2
        range_ratio = (Upper - Lower) / 256.0
        pData = self.m_Voxel

        img_adjusted = (pData - Lower)/range_ratio
        img_adjusted = img_adjusted.clip(0, 255)
        return img_adjusted

    def NumpyArraytoVoxel(self, data):
        self.initialize()

        Dim = data.shape
        self.m_nX = Dim[2]
        self.m_nY = Dim[1]
        self.m_nZ = Dim[0]

        self.m_Voxel = data.astype(np.int16, copy=False)

    def ConvertValue(self, SrcV, TarV):
        idx = self.m_Voxel == SrcV
        self.m_Voxel[idx] = TarV

    def SaveWithoutHeader(self, filename):
        print('success SaveWithoutHeader')
        Save = self.m_Voxel.astype(np.uint8, copy=False)
        print(Save)
        with open(filename, 'wb') as f:
            Save.tofile(f)

    def Normalize(self):
        self.m_Voxel = self.m_Voxel.astype(np.float32, copy=False)
        maxvalue = np.max(self.m_Voxel)
        print(maxvalue)
        self.m_Voxel = self.m_Voxel/maxvalue

    def NormalizeMM(self):
        self.m_Voxel = self.m_Voxel.astype(np.float32, copy=False)
        maxvalue = np.max(self.m_Voxel)
        minvalue = np.min(self.m_Voxel)
        # print minvalue, maxvalue
        diff = maxvalue - minvalue
        self.m_Voxel = (self.m_Voxel - minvalue)/diff

    def AdjustPixelRangeNormalize(self, Upper):
        self.m_Voxel = self.m_Voxel.astype(np.float32, copy=False)
        minvalue = np.min(self.m_Voxel)
        self.m_Voxel = self.m_Voxel.clip(minvalue, Upper)

        diff = Upper - minvalue
        self.m_Voxel = (self.m_Voxel - minvalue)/diff