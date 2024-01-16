from genericpath import isdir
from pathlib import Path
from rembg import remove, new_session
import os
import cv2

bg_path = Path('C:\\Users\\user\\Desktop\\bg') #배경 폴더 위치

obj_orig_path = Path('C:\\Users\\user\\Desktop\\syn_gen_test') #합성물체 이미지 원본 폴더 위치

obj_folder_list = [f for f in obj_orig_path.iterdir() if f.is_dir()] #합성물체 원본 폴더 리스트

result_pre_path = Path(str(obj_orig_path)+'_out') #합성 데이터 통합폴더 생성
result_image_path = Path(result_pre_path/'images') # 누끼이미지 통합폴더 아래 이미지 저장 폴더 생성
result_mask_path = Path(result_pre_path/'masks') #누끼이미지 마스크 파일 폴더 생성

def nukki(): #누끼 이미지 생성
    #누끼이미지 저장 폴더 생성
    if os.path.isdir(result_pre_path)==False:
        os.mkdir(result_pre_path)
    if os.path.isdir(result_image_path)==False:
        os.mkdir(result_image_path)

    session = new_session()

    for ofl in obj_folder_list:
        if os.path.isdir(result_image_path/ofl.stem)==False:
            os.mkdir(result_image_path/ofl.stem)

        for file in ofl.glob('*.png'):
            obj_orig_input_path = str(file)
            nukki_image_output_path = str(result_image_path/ofl.stem / (file.stem + "_out.png"))
            with open(obj_orig_input_path, 'rb') as i:
                with open(nukki_image_output_path, 'wb') as o:
                    input = i.read()
                    output = remove(input, session=session)
                    o.write(output)

        for file in ofl.glob('*.jpg'):
            obj_orig_input_path = str(file)
            nukki_image_output_path = str(result_image_path/ofl.stem / (file.stem + "_out.png"))
            with open(obj_orig_input_path, 'rb') as i:
                with open(nukki_image_output_path, 'wb') as o:
                    input = i.read()
                    output = remove(input, session=session)
                    o.write(output)    

    nukki_image_folder_list = [f for f in result_image_path.iterdir() if f.is_dir()] #누끼이미지 폴더 리스트

    return  nukki_image_folder_list

def bmask_make(): #마스크 이미지 생성
    nukki_image_folder_list = nukki()
    
    if os.path.isdir(result_mask_path)==False:
        os.mkdir(result_mask_path)

    for nifl in nukki_image_folder_list:
        if os.path.isdir(result_mask_path/nifl.stem)==False:
            os.mkdir(result_mask_path/nifl.stem)

        for file in nifl.glob('*.png'):
            nukki_image = cv2.imread(str(file))
            mask_image_output_path = str(result_mask_path/nifl.stem / (file.stem + "_mask.png"))

            gray = cv2.cvtColor(nukki_image, cv2.COLOR_BGR2GRAY) #누끼이미지 색상을 그레이로 변경
            ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY) #물체는 검은색으로, 배경은 흰색으로 변경
            mask_inv = cv2.bitwise_not(mask) # 배경 흰색, 물체 검은색으로 변경

            # cv2.imshow('gray',gray)
            # cv2.imshow('mask',mask) #배경 흰색, 물체 검정
            # cv2.imshow('mask_inv',mask_inv) # 배경 검정, 물체 흰색
            # cv2.waitKeyEx()
            # cv2.destroyAllWindows()

            cv2.imwrite(mask_image_output_path, mask_inv)

if __name__ == "__main__":
    bmask_make()