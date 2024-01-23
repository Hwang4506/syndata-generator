import sys
import os
from PyQt5.QtWidgets import *
from PyQt5 import uic , QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QApplication, QWidget, QTabWidget, QVBoxLayout
from PyQt5.QtGui import *
import tkinter
from tkinter import filedialog
from pathlib import Path
import syn_ui_obj_maker
import syn_ui_data_maker

form_class = uic.loadUiType(str("main.ui"))[0]

class MainWindow(QMainWindow, form_class):

    progressChanged = QtCore.pyqtSignal(int)
    syn_progresschanged = QtCore.pyqtSignal(int)

    def __init__(self) :
        super().__init__()
        self.setupUi(self)

        ########## Obj_Gen Tab ##########
        self.OG_Image_Directory_B.clicked.connect(self.Select_OG_Image_Path)
        self.OG_Image_Directory_LE.setReadOnly(True)
        self.Obj_Run_B.clicked.connect(self.Obj_Gen_Run)
        self.qth_syn = syn_ui_obj_maker.syn_obj_maker(parent = self)
        self.progressChanged.connect(self.Obj_ProgressBar.setValue)

        ########## Image_Gen Tab ##########
        self.Bg_Directory_B.clicked.connect(self.Select_Bg_Path)
        self.Bg_Directory_LE.setReadOnly(True)
        self.Obj_Image_Directory_B.clicked.connect(self.Select_Obj_Image_Path)
        self.Obj_Image_Directory_LE.setReadOnly(True)
        self.Config_Run_B.clicked.connect(self.Config_Run)
        self.sdm = syn_ui_data_maker.syn_data_maker(parent = self)
        self.Syn_Run_B.clicked.connect(self.Syn_Gen_Run)
        self.syn_progresschanged.connect(self.Syn_ProgressBar.setValue)

    ########## Obj_Gen Method ##########
    def Select_OG_Image_Path(self): #합성물체 이미지 폴더 선택
        root = tkinter.Tk()
        root.withdraw()
        
        path = filedialog.askdirectory(parent=root,initialdir = str("./"),title=str("Please select a Image directory") )
        self.OG_Image_Directory_LE.setText(path)
        self.image_pre_path = Path(self.OG_Image_Directory_LE.text())
        self.image_num = sum([len(files) for r, d, files in os.walk(self.OG_Image_Directory_LE.text())])

       
    def Obj_Gen_Run(self):
        try:
            self.Obj_ProgressBar.reset()
            self.qth_syn.obj_orig_path = self.image_pre_path
            self.qth_syn.obj_og_image_num = self.image_num
            self.qth_syn.obj_result_num = 2*self.image_num
            self.qth_syn.start()
        except Exception as e:  
            msg = QMessageBox()
            msg.setWindowTitle(str("Error"))
            msg.setText("경로를 제대로 입력해주세요")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return

    ########## Image_Gen Method ##########
    def Select_Bg_Path(self): #합성물체 이미지 폴더 선택
        root = tkinter.Tk()
        root.withdraw()
        
        path = filedialog.askdirectory(parent=root,initialdir = str("./"),title=str("Please select a Background directory") )
        self.Bg_Directory_LE.setText(path)
        #self.image_pre_path = Path(self.OG_Image_Directory_LE.text())

    def Select_Obj_Image_Path(self): #합성물체 이미지 폴더 선택
        root = tkinter.Tk()
        root.withdraw()
        
        path = filedialog.askdirectory(parent=root,initialdir = str("./"),title=str("Please select a Obj Image directory") )
        self.Obj_Image_Directory_LE.setText(path)
        self.obj_image_path = Path(Path(self.Obj_Image_Directory_LE.text())/"images")
   
    def Config_Run(self):
        try:
            self.obj_folder_list =  [f for f in self.obj_image_path.iterdir() if f.is_dir()] #합성물체 리스트
            self.obj_name_list = []
            for ofl in self.obj_folder_list:
                self.obj_name_list.append(ofl.stem)

            self.Obj_Label_list = [self.Obj_Label_1, self.Obj_Label_2, self.Obj_Label_3, self.Obj_Label_4, self.Obj_Label_5, self.Obj_Label_6,
                            self.Obj_Label_7, self.Obj_Label_8, self.Obj_Label_9, self.Obj_Label_10]
            
            for i in range(len(self.obj_name_list)):
                self.Obj_Label_list[i].setText(self.obj_name_list[i])

        except Exception as e:  
            msg = QMessageBox()
            msg.setWindowTitle(str("Error"))
            msg.setText("경로를 제대로 입력해주세요")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return

    def Syn_Gen_Run(self):
        try:
            self.Syn_ProgressBar.reset()
            self.Obj_dict_list = []
            self.Obj_class_dict = {}
            
            self.Obj_Class_list = [self.Obj_Class_Line_1, self.Obj_Class_Line_2, self.Obj_Class_Line_3, self.Obj_Class_Line_4, self.Obj_Class_Line_5, 
                            self.Obj_Class_Line_6, self.Obj_Class_Line_7, self.Obj_Class_Line_8, self.Obj_Class_Line_9, self.Obj_Class_Line_10]
            
            for i in range(len(self.obj_name_list)):
                self.Obj_dict_list.append([int(self.Obj_Class_list[i].text()), self.Obj_Label_list[i].text()])

            for i in range(len(self.Obj_dict_list)):
                self.Obj_class_dict[int(self.Obj_Class_list[i].text())] = {'folder':self.Obj_Label_list[i].text(), 'longest_min':int(self.Obj_Min.text()), 'longest_max':int(self.Obj_Max.text())}

            self.sdm.obj_dict = self.Obj_class_dict        
            self.sdm.obj_path = Path(self.Obj_Image_Directory_LE.text())
            self.sdm.bg_path = Path(self.Bg_Directory_LE.text())
            self.sdm.Obj_Size_Max = int(self.Obj_Max.text())
            self.sdm.Obj_Size_Min = int(self.Obj_Min.text())
            self.sdm.Bg_Size_Max = int(self.Bg_Max.text())
            self.sdm.Bg_Size_Min = int(self.Bg_Min.text())
            self.sdm.Obj_Number = int(self.Obj_Num.text())
            self.sdm.Syn_Number = int(self.Syn_Num.text())
            self.sdm.Split_Name = self.Split_Name.text()

            if self.sdm.Obj_Size_Min > self.sdm.Obj_Size_Max :
                msg = QMessageBox()
                msg.setWindowTitle(str("Error"))
                msg.setText("합성물체 사이즈를 제대로 입력해주세요")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
                return
            
            if (self.sdm.Obj_Size_Max > self.sdm.Bg_Size_Max) or (self.sdm.Obj_Size_Max > self.sdm.Bg_Size_Min):
                msg = QMessageBox()
                msg.setWindowTitle(str("Error"))
                msg.setText("합성물체와 배경 사이즈를 제대로 입력해주세요")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
                return

            self.sdm.start()
        
        except Exception as e:  
            msg = QMessageBox()
            msg.setWindowTitle(str("Error"))
            msg.setText("설정값을 제대로 입력해주세요")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return

if __name__ == '__main__':
    app = QApplication(sys.argv)
    synWindow = MainWindow()
    synWindow.show()
    app.exec_()

