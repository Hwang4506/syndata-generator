from genericpath import isdir
from pathlib import Path
from rembg import remove, new_session
import os
import cv2
from PyQt5.QtCore import QThread
from PyQt5 import QtCore

class syn_obj_maker(QThread):     
    obj_orig_path = Path() #합성물체 이미지 원본 폴더 위치
    obj_og_image_num = int() #합성물체 이미지 원본 파일 개수
    obj_result_num = int() #합성물체 결과 파일 개수
    numerator = 0
    nukki_image_folder_list = []

    def __init__(self , parent):
        super().__init__()
        self.main = parent

    def nukki(self): #누끼 이미지 생성
        self.obj_folder_list = [f for f in self.obj_orig_path.iterdir() if f.is_dir()] #합성물체 원본 폴더 리스트

        self.result_pre_path = Path(str(self.obj_orig_path)+'_out') #합성 데이터 통합폴더 생성
        self.result_image_path = Path(self.result_pre_path/'images') # 누끼이미지 통합폴더 아래 이미지 저장 폴더 생성
        
        #누끼이미지 저장 폴더 생성
        if os.path.isdir(self.result_pre_path)==False:
            os.mkdir(self.result_pre_path)
        if os.path.isdir(self.result_image_path)==False:
            os.mkdir(self.result_image_path)

        self.session = new_session()

        for ofl in self.obj_folder_list:
            if os.path.isdir(self.result_image_path/ofl.stem)==False:
                os.mkdir(self.result_image_path/ofl.stem)

            # file_list = os.listdir(ofl)
            # denominator = len(file_list)
            # numerator = 0
            # self.main.progressChanged.emit(0)

            for file in ofl.glob('*.png'):
                self.obj_orig_input_path = str(file)
                self.nukki_image_output_path = str(self.result_image_path/ofl.stem / (file.stem + "_out.png"))
                with open(self.obj_orig_input_path, 'rb') as i:
                    with open(self.nukki_image_output_path, 'wb') as o:
                        input = i.read()
                        output = remove(input, session=self.session)
                        o.write(output)
                # numerator +=1
                # self.main.progressChanged.emit(int((numerator/denominator)*100))
                self.numerator +=1
                self.main.progressChanged.emit(int((self.numerator/self.obj_result_num)*100))

            for file in ofl.glob('*.jpg'):
                self.obj_orig_input_path = str(file)
                self.nukki_image_output_path = str(self.result_image_path/ofl.stem / (file.stem + "_out.png"))
                with open(self.obj_orig_input_path, 'rb') as i:
                    with open(self.nukki_image_output_path, 'wb') as o:
                        input = i.read()
                        output = remove(input, session=self.session)
                        o.write(output)
                # numerator +=1
                # self.main.progressChanged.emit(int((numerator/denominator)*100))
                self.numerator +=1
                self.main.progressChanged.emit(int((self.numerator/self.obj_result_num)*100))

        self.nukki_image_folder_list = [f for f in self.result_image_path.iterdir() if f.is_dir()] #누끼이미지 폴더 리스트

    def bmask_make(self): #마스크 이미지 생성
        self.result_mask_path = Path(self.result_pre_path/'masks') #누끼이미지 마스크 파일 폴더 생성
        
        if os.path.isdir(self.result_mask_path)==False:
            os.mkdir(self.result_mask_path)

        for nifl in self.nukki_image_folder_list:
            if os.path.isdir(self.result_mask_path/nifl.stem)==False:
                os.mkdir(self.result_mask_path/nifl.stem)
            
            # self.main.progressChanged.emit(0)

            file_list = os.listdir(nifl)
            denominator = len(file_list)
            numerator = 0

            for file in nifl.glob('*.png'):

                self.nukki_image = cv2.imread(str(file))
                self.mask_image_output_path = str(self.result_mask_path/nifl.stem / (file.stem + "_mask.png"))

                gray = cv2.cvtColor(self.nukki_image, cv2.COLOR_BGR2GRAY) #누끼이미지 색상을 그레이로 변경
                ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY) #물체는 검은색으로, 배경은 흰색으로 변경
                mask_inv = cv2.bitwise_not(mask) # 배경 흰색, 물체 검은색으로 변경

                # cv2.imshow('gray',gray)
                # cv2.imshow('mask',mask) #배경 흰색, 물체 검정
                # cv2.imshow('mask_inv',mask_inv) # 배경 검정, 물체 흰색
                # cv2.waitKeyEx()
                # cv2.destroyAllWindows()

                cv2.imwrite(self.mask_image_output_path, mask_inv)

                # numerator +=1
                # self.main.progressChanged.emit(int((numerator/denominator)*100))
                self.numerator +=1
                self.main.progressChanged.emit(int((self.numerator/self.obj_result_num)*100))

    def run(self) :
        self.numerator = 0
        self.main.progressChanged.emit(0)
        
        self.main.Obj_Run_B.setDisabled(True)
        self.main.OG_Image_Directory_B.setDisabled(True)

        self.nukki()
        self.bmask_make()

        self.main.Obj_Run_B.setEnabled(True)
        self.main.OG_Image_Directory_B.setEnabled(True)
        
        os.startfile(self.result_pre_path)

if __name__ == "__main__":
    syn_obj_maker_test = syn_obj_maker()
    syn_obj_maker_test.nukki()
    syn_obj_maker_test.bmask_make()