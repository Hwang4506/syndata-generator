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
import syn_obj_maker_ui

form_class = uic.loadUiType(str("main.ui"))[0]

class MainWindow(QMainWindow, form_class):

    bg_pre_path = str()

    def __init__(self) :
        super().__init__()
        self.setupUi(self)

        ########## Main Tab ##########
        self.Background_Directory_B.clicked.connect(self.Select_Background_Path)
        self.Background_Directory_LE.setReadOnly(True)
        self.Image_Directory_B.clicked.connect(self.Select_Image_Path)
        self.Image_Directory_LE.setReadOnly(True)
        self.Run_B.clicked.connect(self.Obj_Gen_Run)

    def Select_Background_Path(self):
        root = tkinter.Tk()
        root.withdraw()
        
        path = filedialog.askdirectory(parent=root,initialdir = str("./"),title=str("Please select a Background directory") )
        #path += str("/")
        self.Background_Directory_LE.setText(path)
        # self.bg_pre_path = Path(self.Background_Directory_LE.text())
    

    def Select_Image_Path(self):
        root = tkinter.Tk()
        root.withdraw()
        
        path = filedialog.askdirectory(parent=root,initialdir = str("./"),title=str("Please select a Image directory") )
        #path += str("/")
        self.Image_Directory_LE.setText(path)
        self.image_pre_path = Path(self.Image_Directory_LE.text())
       

    def Obj_Gen_Run(self):
        self.syn_obj_cls = syn_obj_maker_ui.syn_obj_maker()
        self.syn_obj_cls.obj_orig_path = self.image_pre_path
        self.syn_obj_cls.nukki()
        self.syn_obj_cls.bmask_make()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    synWindow = MainWindow()
    synWindow.show()
    app.exec_()

