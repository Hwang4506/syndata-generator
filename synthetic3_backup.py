import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import albumentations as A
import time
from tqdm import tqdm
import sys
import random

# path 설정
obj_dict = {
    4: {'folder': "fire2", 'longest_min': 150, 'longest_max': 200},
    #2: {'folder': "flame", 'longest_min': 150, 'longest_max': 200},
} #longest_min : 합성 물체 이미지와 배경 이미지의 h,w 크기의 최소값(default=30), longest_max : 합성 물체 이미지와 배경 이미지의 h,w 크기의 최대값

PATH_MAIN = "C:\\Users\\user\\Desktop\\syn_test\\"

obj_list = list(obj_dict.keys())

# radkey = random.choice(obj_list)

# print(radkey)

# print(obj_list)
# print(type(obj_list))

for k, _ in obj_dict.items():
    folder_name = obj_dict[k]['folder']
        
    files_imgs = sorted(os.listdir(os.path.join(PATH_MAIN, folder_name, 'images')))
    files_imgs = [os.path.join(PATH_MAIN, folder_name, 'images', f) for f in files_imgs]
        
    files_masks = sorted(os.listdir(os.path.join(PATH_MAIN, folder_name, 'masks')))
    files_masks = [os.path.join(PATH_MAIN, folder_name, 'masks', f) for f in files_masks]
        
    obj_dict[k]['images'] = files_imgs
    obj_dict[k]['masks'] = files_masks
    
print("The first five files from the sorted list of fire images:", obj_dict[4]['images'][:5])
print("\nThe first five files from the sorted list of fire masks:", obj_dict[4]['masks'][:5])

files_bg_imgs = sorted(os.listdir(os.path.join(PATH_MAIN, 'bg')))
files_bg_imgs = [os.path.join(PATH_MAIN, 'bg', f) for f in files_bg_imgs]

'''
# 노이즈 추가
files_bg_noise_imgs = sorted(os.listdir(os.path.join(PATH_MAIN, "bg_noise", "images")))
files_bg_noise_imgs = [os.path.join(PATH_MAIN, "bg_noise", "images", f) for f in files_bg_noise_imgs]
files_bg_noise_masks = sorted(os.listdir(os.path.join(PATH_MAIN, "bg_noise", "masks")))
files_bg_noise_masks = [os.path.join(PATH_MAIN, "bg_noise", "masks", f) for f in files_bg_noise_masks]
'''

print("\nThe first five files from the sorted list of background images:", files_bg_imgs[:5])
#print("\nThe first five files from the sorted list of background noise images:", files_bg_noise_imgs[:5])
#print("\nThe first five files from the sorted list of background noise masks:", files_bg_noise_masks[:5])


# bynary mask 생성 
def get_img_and_mask(img_path, mask_path): 

    img = cv2.imread(img_path)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    mask = cv2.imread(mask_path)
    #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    
    mask_b = mask[:,:,0] == 0 # This is boolean mask
    mask = mask_b.astype(np.uint8) # This is binary mask
    
    return img, mask

""" 
# bynary mask 생성 테스트
img_path = obj_dict[1]['images'][0]
mask_path = obj_dict[1]['masks'][0]

img, mask = get_img_and_mask(img_path, mask_path)

print("\nImage file:", img_path)
print("Mask file:", mask_path)
print("\nShape of the image of the object:", img.shape)
print("Shape of the binary mask:", mask.shape)

# fig, ax = plt.subplots(1, 2, figsize=(16, 7))
# ax[0].imshow(img)
# ax[0].set_title('Object', fontsize=18)
# ax[1].imshow(mask)
# ax[1].set_title('Binary mask', fontsize=18)
cv2.imshow('image', img)
cv2.imshow('binary mask', mask)
cv2.waitKey()
cv2.destroyAllWindows()
 """

# 배경 이미지 사이즈 조절
def resize_img(img, desired_max, desired_min=None): #배경 이미지 리사이징 최소값, 최대값 지정(w,h중 큰값이 최대값으로 지정)
    # 최소값을 지정하지 않으면 w,h 중 작은값은 원본 비율대로 적용, 최소값을 지정하면 지정한 최소값으로 적용
   
    h, w = img.shape[0], img.shape[1]
    
    longest, shortest = max(h, w), min(h, w)
    longest_new = desired_max
    if desired_min:
        shortest_new = desired_min
    else:
        shortest_new = int(shortest * (longest_new / longest))
    
    if h > w:
        h_new, w_new = longest_new, shortest_new
    else:
        h_new, w_new = shortest_new, longest_new
        
    transform_resize = A.Compose([
        A.Sequential([
        A.Resize(h_new, w_new, interpolation=1, always_apply=False, p=1)
        ], p=1)
    ])

    transformed = transform_resize(image=img)
    img_r = transformed["image"]
        
    return img_r

""" 
# 배경 이미지 사이즈 조절 테스트
img_bg_path = files_bg_imgs[0]
img_bg = cv2.imread(img_bg_path)
#img_bg = cv2.cvtColor(img_bg, cv2.COLOR_BGR2RGB)

img_bg_resized_1 = resize_img(img_bg, desired_max=800, desired_min=None)
img_bg_resized_2 = resize_img(img_bg, desired_max=800, desired_min=800)

print("Shape of the original background image:", img_bg.shape)

print("Shape of the resized background image (desired_max=800, desired_min=None):", img_bg_resized_1.shape)
print("Shape of the resized background image (desired_max=800, desired_min=800):", img_bg_resized_2.shape)

#fig, ax = plt.subplots(1, 2, figsize=(16, 7))
#ax[0].imshow(img_bg_resized_1)
cv2.imshow('re1', img_bg_resized_1)
#ax[0].set_title('Resized (desired_max=1920, desired_min=None)', fontsize=18)
#ax[1].imshow(img_bg_resized_2)
cv2.imshow('re2', img_bg_resized_2)
#ax[1].set_title('Resized (desired_max=1920, desired_min=1080)', fontsize=18)
cv2.waitKey()
cv2.destroyAllWindows()
 """

# 대상 이미지 사이즈 조절
def resize_transform_obj(img, mask, longest_min, longest_max, transforms=False): #합성물체 이미지 리사이징 최소값, 최대값 지정 / 합성물체 로테이트, 밝기조절 설정
    #이미지 리사이징 최소값과 최대값 사이의 랜덤값으로 합성물체 리사이징(w,h중 큰값은 랜덤값이 되고 작은값은 원본 비율에 맞춰 생성 / w,h중 큰값으로 지정 되는것도 랜덤)
   
    h, w = mask.shape[0], mask.shape[1]
    
    longest, shortest = max(h, w), min(h, w)
    longest_new = np.random.randint(longest_min, longest_max)
    shortest_new = int(shortest * (longest_new / longest))
    
    if h > w:
        h_new, w_new = longest_new, shortest_new
    else:
        h_new, w_new = shortest_new, longest_new
        
    transform_resize = A.Resize(h_new, w_new, interpolation=1, always_apply=False, p=1)

    transformed_resized = transform_resize(image=img, mask=mask)
    img_t = transformed_resized["image"]
    mask_t = transformed_resized["mask"]
        
    if transforms:
        transformed = transforms(image=img_t, mask=mask_t)
        img_t = transformed["image"]
        mask_t = transformed["mask"]
        
    return img_t, mask_t

transforms_bg_obj = A.Compose([
    A.RandomRotate90(p=1),
    A.ColorJitter(brightness=0.3,
                  contrast=0.3,
                  saturation=0.3,
                  hue=0.07,
                  always_apply=False,
                  p=1),
    A.Blur(blur_limit=(3,15),
           always_apply=False,
           p=0.5)
])

transforms_obj = A.Compose([ #transforms=transforms_obj 일때, 실행 작업
    A.RandomRotate90(p=1), #로테이트 90 or 180 or 270 랜덤
    A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.2),
                               contrast_limit=0.1,
                               brightness_by_max=True,
                               always_apply=False,
                               p=1)
])

""" 
# 대상 이미지 사이즈 조절 테스트
img_path = obj_dict[1]['images'][0]
mask_path = obj_dict[1]['masks'][0]

img, mask = get_img_and_mask(img_path, mask_path)

img_t, mask_t = resize_transform_obj(img,
                                     mask,
                                     longest_min=500,
                                     longest_max=600,
                                     transforms=transforms_obj)

print("Shape of the original image:", img.shape)
print("Shape of the image of the transformed object:", img_t.shape)
print("Shape of the transformed binary mask:", mask_t.shape)
print("\n")

#fig, ax = plt.subplots(1, 2, figsize=(16, 7))
#ax[0].imshow(img_t)
#ax[0].set_title('Transformed object', fontsize=18)
#ax[1].imshow(mask_t)
#ax[1].set_title('Transformed binary mask', fontsize=18)
cv2.imshow('Transformed object', img_t)
cv2.imshow('Transformed binary mask', mask_t)
cv2.waitKey()
cv2.destroyAllWindows()
 """


# 배경 합성
def add_obj(img_comp, mask_comp, img, mask, x, y, idx):
    '''
    img_comp - (리사이징한)배경 이미지
    mask_comp - 배경 이미지를 모두 0으로 채운 binary mask 이미지(img_comp를 np.zeros해서 생성)
    img - 합성물체 이미지
    mask - 합성물체 binary mask 이미지
    x, y - 합성물체가 배치되는 오른쪽 위 x,y 값
    idx - 합성물체 id
    return값 - 합성이미지, 합성이미지의 binary mask 이미지, 마지막에 추가된 물체의 mask
    '''
    h_comp, w_comp = img_comp.shape[0], img_comp.shape[1]
    
    h, w = img.shape[0], img.shape[1]
    
    x = x - int(w/2)
    y = y - int(h/2)
    
    mask_b = mask == 1
    mask_rgb_b = np.stack([mask_b, mask_b, mask_b], axis=2)
    
    if x >= 0 and y >= 0: #합성물체가 배경안에 완전히 들어올때
    
        h_part = h - max(0, y+h-h_comp) # h_part - part of the image which gets into the frame of img_comp along y-axis
        w_part = w - max(0, x+w-w_comp) # w_part - part of the image which gets into the frame of img_comp along x-axis

        img_comp[y:y+h_part, x:x+w_part, :] = img_comp[y:y+h_part, x:x+w_part, :] * ~mask_rgb_b[0:h_part, 0:w_part, :] + (img * mask_rgb_b)[0:h_part, 0:w_part, :]
        mask_comp[y:y+h_part, x:x+w_part] = mask_comp[y:y+h_part, x:x+w_part] * ~mask_b[0:h_part, 0:w_part] + (idx * mask_b)[0:h_part, 0:w_part]
        mask_added = mask[0:h_part, 0:w_part]
        
    elif x < 0 and y < 0: #합성물체의 가로와 세로 일부분이 배경 밖으로 나갈때
        
        h_part = h + y
        w_part = w + x
        
        img_comp[0:0+h_part, 0:0+w_part, :] = img_comp[0:0+h_part, 0:0+w_part, :] * ~mask_rgb_b[h-h_part:h, w-w_part:w, :] + (img * mask_rgb_b)[h-h_part:h, w-w_part:w, :]
        mask_comp[0:0+h_part, 0:0+w_part] = mask_comp[0:0+h_part, 0:0+w_part] * ~mask_b[h-h_part:h, w-w_part:w] + (idx * mask_b)[h-h_part:h, w-w_part:w]
        mask_added = mask[h-h_part:h, w-w_part:w]
        
    elif x < 0 and y >= 0: #합성물체의 세로는 배경 안으로 완전히 들어가고 합성물체의 가로는 일부분이 배경 밖으로 나갈때
        
        h_part = h - max(0, y+h-h_comp)
        w_part = w + x
        
        img_comp[y:y+h_part, 0:0+w_part, :] = img_comp[y:y+h_part, 0:0+w_part, :] * ~mask_rgb_b[0:h_part, w-w_part:w, :] + (img * mask_rgb_b)[0:h_part, w-w_part:w, :]
        mask_comp[y:y+h_part, 0:0+w_part] = mask_comp[y:y+h_part, 0:0+w_part] * ~mask_b[0:h_part, w-w_part:w] + (idx * mask_b)[0:h_part, w-w_part:w]
        mask_added = mask[0:h_part, w-w_part:w]
        
    elif x >= 0 and y < 0: #합성물체의 가로는 배경 안으로 완전히 들어가고 합성물체의 세로는 일부분이 배경 밖으로 나갈때
        
        h_part = h + y
        w_part = w - max(0, x+w-w_comp)
        
        img_comp[0:0+h_part, x:x+w_part, :] = img_comp[0:0+h_part, x:x+w_part, :] * ~mask_rgb_b[h-h_part:h, 0:w_part, :] + (img * mask_rgb_b)[h-h_part:h, 0:w_part, :]
        mask_comp[0:0+h_part, x:x+w_part] = mask_comp[0:0+h_part, x:x+w_part] * ~mask_b[h-h_part:h, 0:w_part] + (idx * mask_b)[h-h_part:h, 0:w_part]
        mask_added = mask[h-h_part:h, 0:w_part]

    return img_comp, mask_comp, mask_added

""" 
# 배경 합성 테스트
img_bg_path = files_bg_imgs[0]
img_bg = cv2.imread(img_bg_path)
#img_bg = cv2.cvtColor(img_bg, cv2.COLOR_BGR2RGB)
re_img_bg = resize_img(img_bg, desired_max=900, desired_min=900)


h, w = re_img_bg.shape[0], re_img_bg.shape[1]
mask_comp = np.zeros((h,w), dtype=np.uint8)

img_path = obj_dict[1]['images'][0]
mask_path = obj_dict[1]['masks'][0]
img, mask = get_img_and_mask(img_path, mask_path)
img_t, mask_t = resize_transform_obj(img,
                                     mask,
                                     longest_min=300,
                                     longest_max=400,
                                     transforms=transforms_obj)

img_comp, mask_comp, res = add_obj(re_img_bg, mask_comp, img_t, mask_t, x=400, y=300, idx=1)

print("Shape of the background image:", re_img_bg.shape)
print("Shape of the object image:", img_t.shape)
#fig, ax = plt.subplots(1, 2, figsize=(16, 7))
#ax[0].imshow(img_comp)
#ax[0].set_title('Composition', fontsize=18)
#ax[1].imshow(mask_comp)
#ax[1].set_title('Composition mask', fontsize=18)
cv2.imshow('composition', img_comp)
cv2.imshow('composition mask', mask_comp)
cv2.waitKey()
cv2.destroyAllWindows()
 """

# 합성물체 중첩 영역 확인 #새로운 합성물체가 기존 합성물체와 중첩이 overlap_degree 이상일 경우 False, 이하일 경우 True 반환
def check_areas(mask_comp, obj_areas, overlap_degree=0.3): 
    """ 
    mask_comp - 합성 이미지의 mask
    obj_areas - 합성 물체들의 mask list
    overlap_degree - 중첩 영역 threshold
       """
    obj_ids = np.unique(mask_comp).astype(np.uint8)[1:-1]
    masks = mask_comp == obj_ids[:, None, None]
    
    ok = True
    
    if len(np.unique(mask_comp)) != np.max(mask_comp) + 1:
        ok = False
        return ok
    
    for idx, mask in enumerate(masks):
        if np.count_nonzero(mask) / obj_areas[idx] < 1 - overlap_degree:
            ok = False
            break
            
    return ok   

# 합성 데이터 생성
def create_composition(img_comp_bg,
                       max_objs=15,
                       overlap_degree=0.2,
                       max_attempts_per_obj=10):
    """ 
     img_comp_bg - (리사이징된)배경 이미지
     max_objs - 합성물체 최대 개수
     overlap_degree - 합성물체 중첩 threshold
     max_attempts_per_obj - 합성물체가 중첩될 경우 새로 합성을 시도하는 횟수
     return값 - 합성 이미지, 합성 이미지의 mask, 합성물체 class list(합성물체의 class의 숫자와 이름은 obj_dict에 정의), 합성물체들의 mask list
        """

    img_comp = img_comp_bg.copy()
    h, w = img_comp.shape[0], img_comp.shape[1]
    mask_comp = np.zeros((h,w), dtype=np.uint8)
    
    obj_areas = []
    labels_comp = []
    num_objs = np.random.randint(max_objs) + 2
    
    i = 1
    
    for _ in range(1, num_objs):

        obj_idx = np.random.randint(len(obj_dict)) + 1
        
        for _ in range(max_attempts_per_obj):
            obj_number = random.choice(obj_list)

            imgs_number = len(obj_dict[obj_number]['images'])
            idx = np.random.randint(imgs_number)
            img_path = obj_dict[obj_number]['images'][idx]
            mask_path = obj_dict[obj_number]['masks'][idx]
            img, mask = get_img_and_mask(img_path, mask_path)

            x, y = np.random.randint(w), np.random.randint(h)
            longest_min = obj_dict[obj_number]['longest_min']
            longest_max = obj_dict[obj_number]['longest_max']
            img, mask = resize_transform_obj(img,
                                             mask,
                                             longest_min,
                                             longest_max,
                                             transforms=transforms_obj)

            if i == 1:
                img_comp, mask_comp, mask_added = add_obj(img_comp,
                                                          mask_comp,
                                                          img,
                                                          mask,
                                                          x,
                                                          y,
                                                          i)
                obj_areas.append(np.count_nonzero(mask_added))
                labels_comp.append(obj_number)
                i += 1
                break
            else:        
                img_comp_prev, mask_comp_prev = img_comp.copy(), mask_comp.copy()
                img_comp, mask_comp, mask_added = add_obj(img_comp,
                                                          mask_comp,
                                                          img,
                                                          mask,
                                                          x,
                                                          y,
                                                          i)
                ok = check_areas(mask_comp, obj_areas, overlap_degree)
                if ok:
                    obj_areas.append(np.count_nonzero(mask_added))
                    labels_comp.append(obj_number)
                    i += 1
                    break
                else:
                    img_comp, mask_comp = img_comp_prev.copy(), mask_comp_prev.copy()        
        
    return img_comp, mask_comp, labels_comp, obj_areas

""" 
# 합성 데이터 생성 테스트
img_bg_path = files_bg_imgs[0]
img_bg = cv2.imread(img_bg_path)
img_bg = cv2.cvtColor(img_bg, cv2.COLOR_BGR2RGB)
re_img_bg = resize_img(img_bg, desired_max=800, desired_min=800)
img_comp, mask_comp, labels_comp, obj_areas = create_composition(re_img_bg,
                                                                 max_objs=15,
                                                                 overlap_degree=0.2,
                                                                 max_attempts_per_obj=10)
# 합성 이미지 결과 출력
#cv2.imshow('sy_image', img_comp)

# 합성된 물체 class 출력
print("Labels (classes of the objects) on the composition in order of object's addition:", labels_comp)

# 합성 물체들 간 중첩 정도 출력
obj_ids = np.unique(mask_comp).astype(np.uint8)[1:]
masks = mask_comp == obj_ids[:, None, None]

print("Degree of how much area of each object is overlapped:")

for idx, mask in enumerate(masks):
    print(np.count_nonzero(mask) / obj_areas[idx])

# 합성 데이터에 바운딩 박스 표시
colors = {1: (255,0,0), 2: (0,255,0), 3: (0,0,255)} #바운딩 박스 색깔 조절

img_comp_bboxes = img_comp.copy()

obj_ids = np.unique(mask_comp).astype(np.uint8)[1:]
masks = mask_comp == obj_ids[:, None, None]

for i in range(len(obj_ids)):
    pos = np.where(masks[i])
    xmin = np.min(pos[1])
    xmax = np.max(pos[1])
    ymin = np.min(pos[0])
    ymax = np.max(pos[0])
    img_comp_bboxes = cv2.rectangle(img_comp_bboxes,
                                    (xmin, ymin),
                                    (xmax,ymax),
                                    colors[labels_comp[i]], 
                                    6) #바운딩 박스 두께 조절
    
#plt.figure(figsize=(40,40))
#plt.imshow(img_comp_bboxes)  
cv2.imshow('sy_bbox_image', img_comp_bboxes)
cv2.waitKey()
cv2.destroyAllWindows()
 """

# 라벨링 파일 생성
def create_yolo_annotations(mask_comp, labels_comp):
    """ 
    mask_comp - 합성 이미지의 mask
    lables_comp - 합성물체 class list
       """
    comp_w, comp_h = mask_comp.shape[1], mask_comp.shape[0]
    
    obj_ids = np.unique(mask_comp).astype(np.uint8)[1:]
    masks = mask_comp == obj_ids[:, None, None]

    annotations_yolo = []
    for i in range(len(labels_comp)):
        pos = np.where(masks[i])
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])

        xc = (xmin + xmax) / 2
        yc = (ymin + ymax) / 2
        w = xmax - xmin
        h = ymax - ymin

        annotations_yolo.append([labels_comp[i],
                                 round(xc/comp_w, 5),
                                 round(yc/comp_h, 5),
                                 round(w/comp_w, 5),
                                 round(h/comp_h, 5)])

    return annotations_yolo

# 텍스트 파일 저장
def printsave(*a):
    file = open('C:\\Users\\user\\Desktop\\syn_test\\result.txt','a')
    print(*a,file=file)
    file.close()

# 라벨링 파일 생성 테스트
img_bg_path = files_bg_imgs[0]
img_bg = cv2.imread(img_bg_path)
#img_bg = cv2.cvtColor(img_bg, cv2.COLOR_BGR2RGB)
re_img_bg = resize_img(img_bg, desired_max=800, desired_min=800)
img_comp, mask_comp, labels_comp, obj_areas = create_composition(re_img_bg,
                                                                 max_objs=15,
                                                                 overlap_degree=0.2,
                                                                 max_attempts_per_obj=10)


#colors = {1: (255,0,0), 2: (0,255,0), 3: (0,0,255)}
colors = {}
for i in range(0,50):
    color_seed = i
    random.seed(color_seed)
    colors[i] = (random.randrange(0,255+1),random.randrange(0,255+1),random.randrange(0,255+1))

img_comp_bboxes = img_comp.copy()

obj_ids = np.unique(mask_comp).astype(np.uint8)[1:]
masks = mask_comp == obj_ids[:, None, None]

for i in range(len(obj_ids)):
    pos = np.where(masks[i])
    xmin = np.min(pos[1])
    xmax = np.max(pos[1])
    ymin = np.min(pos[0])
    ymax = np.max(pos[0])
    img_comp_bboxes = cv2.rectangle(img_comp_bboxes,
                                    (xmin, ymin),
                                    (xmax,ymax),
                                    colors[labels_comp[i]],
                                    6)

cv2.imshow('sy_bbox_image', img_comp_bboxes)
cv2.imwrite('C:\\Users\\user\\Desktop\\syn_test\\result.jpg', img_comp)
cv2.waitKey()
cv2.destroyAllWindows()

annotations_yolo = create_yolo_annotations(mask_comp, labels_comp)
for i in range(len(annotations_yolo)):
    print(' '.join(str(el) for el in annotations_yolo[i]))
for i in range(len(annotations_yolo)):
    printsave(' '.join(str(el) for el in annotations_yolo[i]))



#합성 데이터 셋 생성
def generate_dataset(imgs_number, folder, split='train'):
    time_start = time.time()
    for j in tqdm(range(imgs_number)):        
        img_comp, mask_comp, labels_comp, _ = create_composition(img_comp_bg,
                                                                 max_objs=15,
                                                                 overlap_degree=0.2,
                                                                 max_attempts_per_obj=10)

        img_comp = cv2.cvtColor(img_comp, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(folder, split, 'images/{}.jpg').format(j), img_comp)

        annotations_yolo = create_yolo_annotations(mask_comp, labels_comp)
        for i in range(len(annotations_yolo)):
            with open(os.path.join(folder, split, 'labels/{}.txt').format(j), "a") as f:
                f.write(' '.join(str(el) for el in annotations_yolo[i]) + '\n')
                
    time_end = time.time()
    time_total = round(time_end - time_start)
    time_per_img = round((time_end - time_start) / imgs_number, 1)
    
    print("Generation of {} synthetic images is completed. It took {} seconds, or {} seconds per image".format(imgs_number, time_total, time_per_img))
    print("Images are stored in '{}'".format(os.path.join(folder, split, 'images')))
    print("Annotations are stored in '{}'".format(os.path.join(folder, split, 'labels')))

    generate_dataset(10, folder='dataset', split='train')
