import cv2
import numpy as np
from pathlib import Path
import os
from genericpath import isdir
import bgremove

result_mask_path = Path(bgremove.result_pre_path/'mask') #누끼이미지 마스크 파일 폴더 생성

def bmask_make():
    if os.path.isdir(result_mask_path)==False:
        os.mkdir(result_mask_path)

    for file in image_path.glob('*.png'):
        or_image = cv2.imread(str(file))
        output_path = str(result_mask_path / (file.stem + "_mask.png"))

        gray = cv2.cvtColor(or_image, cv2.COLOR_BGR2GRAY) #누끼이미지 색상을 그레이로 변경
        ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY) #물체는 검은색으로, 배경은 흰색으로 변경
        mask_inv = cv2.bitwise_not(mask) # 배경 흰색, 물체 검은색으로 변경

        # cv2.imshow('gray',gray)
        # cv2.imshow('mask',mask) #배경 흰색, 물체 검정
        # cv2.imshow('mask_inv',mask_inv) # 배경 검정, 물체 흰색
        # cv2.waitKeyEx()
        # cv2.destroyAllWindows()

        cv2.imwrite(output_path, mask_inv)

if __name__ == "__main__":
    bmask_make()
