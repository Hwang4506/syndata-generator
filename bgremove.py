from genericpath import isdir
from pathlib import Path
from rembg import remove, new_session
import os

def nukki():
    image_path = Path('C:\\Users\\user\\Desktop\\bg_test') #합성물체 이미지 위치
    #누끼이미지 저장 폴더 생성
    result_path1 = Path(str(image_path)+'_out') #image_path 상위 디렉토리에 누끼이미지 통합폴더 생성
    if os.path.isdir(result_path1)==False:
        os.mkdir(result_path1)
    result_path2 = Path(result_path1/'images') # 누끼이미지 통합폴더 아래 이미지 저장 폴더 생성
    if os.path.isdir(result_path2)==False:
        os.mkdir(result_path2)
    
    session = new_session()

    for file in image_path.glob('*.png'):
        input_path = str(file)
        output_path = str(result_path2 / (file.stem + "_out.png"))
        with open(input_path, 'rb') as i:
            with open(output_path, 'wb') as o:
                input = i.read()
                output = remove(input, session=session)
                o.write(output)

    for file in image_path.glob('*.jpg'):
        input_path = str(file)
        output_path = str(result_path2 / (file.stem + "_out.png"))
        with open(input_path, 'rb') as i:
            with open(output_path, 'wb') as o:
                input = i.read()
                output = remove(input, session=session)
                o.write(output)
    

if __name__ == "__main__":
    nukki()