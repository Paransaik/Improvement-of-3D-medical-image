import vtk

class VolumeRendering:
    def __init__(self, path):
        self.reader = vtk.vtkDICOMImageReader()
        self.reader.SetDirectoryName(r'{}'.format(path))
        self.reader.Update()

        self.c_f, self.o_f = vtk.vtkColorTransferFunction(), vtk.vtkPiecewiseFunction()
        self.p = vtk.vtkVolumeProperty()
        self.m, self.v = vtk.vtkSmartVolumeMapper(), vtk.vtkVolume()

        self.ren = vtk.vtkRenderer()
        self.renWin = vtk.vtkRenderWindow()
        self.iren = vtk.vtkRenderWindowInteractor()
        self.iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera()) # InteractorStyle을 TrackballCamera로 바꾸니 확실히 움직이며 확인하기 편해졌다.

        self.mapper()
        self.property()
        self.actor()
        self.rendering()

    def property(self):
        opacityWindow = 4096  # 값 바꿔가며 어떤 걸 변경하는 값인지 확인해보기
        opacityLevel = 2048  # ...

        self.c_f.AddRGBSegment(0.0, 1.0, 1.0, 1.0, 255.0, 1.0, 1.0, 1.0)
        self.o_f.AddSegment(opacityLevel - 0.5 * opacityWindow, 0.0, opacityLevel + 0.5 * opacityWindow, 1.0)

        self.p.SetColor(self.c_f)
        self.p.SetScalarOpacity(self.o_f)

    def mapper(self):
        self.m.SetInputConnection(self.reader.GetOutputPort())
        self.m.SetBlendModeToMaximumIntensity()
    
    def actor(self):
        self.v.SetMapper(self.m)
        self.v.SetProperty(self.p)
    
    def rendering(self):
        self.renWin.AddRenderer(self.ren)
        self.renWin.SetSize(600, 600)
        self.iren.SetRenderWindow(self.renWin)

        self.ren.AddViewProp(self.v)
        self.renWin.Render()
        self.iren.Initialize()
        self.iren.Start()

class SliceRendering:
    def __init__(self, path):
        self.reader = vtk.vtkDICOMImageReader()
        self.reader.SetFileName(r'{}'.format(path))
        self.reader.Update()

        self.m, self.a = vtk.vtkImageResliceMapper(), vtk.vtkImageSlice()
        self.p = vtk.vtkVolumeProperty()
        self.plane = vtk.vtkPlane()

        self.ren = vtk.vtkRenderer()
        self.renWin = vtk.vtkRenderWindow()
        self.iren = vtk.vtkRenderWindowInteractor()

        self.source()
        self.mapper()
        self.actor()
        self.rendering()

    def source(self):
        # data extent(데이터 범위)
        (xMin, xMax, yMin, yMax, zMin, zMax) = self.reader.GetExecutive().GetWholeExtent(self.reader.GetOutputInformation(0))
        (xSpacing, ySpacing, zSpacing) = self.reader.GetOutput().GetSpacing() 
        (x0, y0, z0) = self.reader.GetOutput().GetOrigin() 

        # center of volume
        center = [x0 + xSpacing * 0.5 * (xMin + xMax),
                y0 + ySpacing * 0.5 * (yMin + yMax),
                z0 + zSpacing * 0.5 * (zMin + zMax)]

        # set cutting plane
        self.plane.SetOrigin(center) 
        self.plane.SetNormal(0, 0, 1) 
        # viewUp = [0, -1, 0] 
        # i, j, k = 0, 0, -zMax 

    def mapper(self):
        # Mapper
        self.m.SetInputConnection(self.reader.GetOutputPort())
        self.m.SetSlicePlane(self.plane)  

    def actor(self):
        # Actor
        self.a.SetMapper(self.m)

    def rendering(self):
        self.renWin.AddRenderer(self.ren)
        self.renWin.SetSize(600, 600)
        self.iren.SetRenderWindow(self.renWin)
        
        self.ren.AddViewProp(self.a)
        
        self.renWin.Render()
        self.iren.Initialize()
        self.iren.Start()