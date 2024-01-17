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

form_class = uic.loadUiType(str("main.ui"))[0]

class MainWindow(QMainWindow, form_class):

    progressChanged = QtCore.pyqtSignal(int)

    def __init__(self) :
        super().__init__()
        self.setupUi(self)

        ########## Main Tab ##########
        self.Obj_Image_Directory_B.clicked.connect(self.Select_Image_Path)
        self.Obj_Image_Directory_LE.setReadOnly(True)
        self.Obj_Run_B.clicked.connect(self.Obj_Gen_Run)
        self.qth_syn = syn_ui_obj_maker.syn_obj_maker(parent = self)
        self.progressChanged.connect(self.Obj_ProgressBar.setValue)
    
    def Select_Image_Path(self): #합성물체 이미지 폴더 선택
        root = tkinter.Tk()
        root.withdraw()
        
        path = filedialog.askdirectory(parent=root,initialdir = str("./"),title=str("Please select a Image directory") )
        self.Obj_Image_Directory_LE.setText(path)
        self.image_pre_path = Path(self.Obj_Image_Directory_LE.text())
       

    def Obj_Gen_Run(self):
        self.Obj_ProgressBar.reset()
        self.qth_syn.obj_orig_path = self.image_pre_path
        self.qth_syn.start()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    synWindow = MainWindow()
    synWindow.show()
    app.exec_()

