from PIL import Image
import os

# path = r'C:/Users/Hp/Desktop/monodepth2/data_from_lidar/lidar/train/0ef28d5c-ae34-370b-99e7-6709e1c4b929/stereo_front_left_rect_disparity/'
# path_out = r'C:/Users/Hp/Desktop/monodepth2/assets/images/'
# for file in os.listdir(path):
#     img = Image.open(path + file).convert('LA')
#     file_name, file_ext = os.path.splitext(file)
#     img.save(path_out + file + '.png')

import numpy as np
import cv2
from matplotlib import pyplot as plt
path_left = r'C:/Users/Hp/Desktop/monodepth2/data_from_lidar/photo/train/0ef28d5c-ae34-370b-99e7-6709e1c4b929/stereo_front_left_rect/'
path_right = r'C:/Users/Hp/Desktop/monodepth2/data_from_lidar/photo/train/0ef28d5c-ae34-370b-99e7-6709e1c4b929/stereo_front_right_rect/'
image_to_disp = r'C:/Users/Hp/Desktop/monodepth2/assets/disparity/image/'
disp_result = r'C:/Users/Hp/Desktop/monodepth2/assets/disparity/sum/'

# #Забираю из папки с датасетом train первой по счету все изображения левой камеры
count = 0
for file in os.listdir(path_left):
    img = Image.open(path_left + file)
    # file_name, file_ext = os.path.splitext(file)
    img.save(image_to_disp + 'left_' + str(count) + '.jpg')
    # os.replace(path_left + file_name + '.jpg', image_to_disp + file_name + '.jpg')
    # os.rename(image_to_disp + file_name + '.jpg', image_to_disp + 'left_' + str(count) + '.jpg')
    count += 1

#Забираю из папки с датасетом train первой по счету все изображения правой камеры
count = 0
for file in os.listdir(path_right):
    img = Image.open(path_right + file)
    # file_name, file_ext = os.path.splitext(file)
    img.save(image_to_disp + 'right_' + str(count) + '.jpg')
    # os.replace(path_right + file_name + '.jpg', image_to_disp + file_name + '.jpg')
    # os.rename(image_to_disp + file_name + '.jpg', image_to_disp + 'right_' + str(count) + '.jpg')
    count += 1

count = 0
for file in os.listdir(image_to_disp):
    imgL = cv2.imread(image_to_disp + 'left_' + str(count) + '.jpg', 0)
    imgR = cv2.imread(image_to_disp + 'right_' + str(count) + '.jpg', 0)
    # stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    # stereo = cv2.StereoBM_create()
    # disparity = stereo.compute(imgL,imgR)


    block_size = 11
    min_disp = -128
    max_disp = 128

    num_disp = max_disp - min_disp

    uniquenessRatio = 5

    speckleWindowSize = 200

    speckleRange = 2
    disp12MaxDiff = 0

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        uniquenessRatio=uniquenessRatio,
        speckleWindowSize=speckleWindowSize,
        speckleRange=speckleRange,
        disp12MaxDiff=disp12MaxDiff,
        P1=8 * 1 * block_size * block_size,
        P2=32 * 1 * block_size * block_size,
    )
    disparity_SGBM = stereo.compute(imgL, imgR)

    # Normalize the values to a range from 0..255 for a grayscale image
    disparity_SGBM = cv2.normalize(disparity_SGBM, disparity_SGBM, alpha=255,
                                beta=0, norm_type=cv2.NORM_MINMAX)
    disparity_SGBM = np.uint8(disparity_SGBM)
    # Можно запускать карту глубины без фильтра по disparity_SGBM

    wsize=31
    max_disp = 128
    sigma = 1.5 #Размытость default = 1.5 (>1)
    lmbda = 8000.0
    # left_matcher = cv2.StereoBM_create(max_disp, wsize)
    left_matcher = stereo
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    left_disp = left_matcher.compute(imgL, imgR)
    right_disp = right_matcher.compute(imgR,imgL)

    # Now create DisparityWLSFilter
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    filtered_disp = wls_filter.filter(left_disp, imgL, disparity_map_right=right_disp)


    image_name = 'disp_' + str(count)
    plt.imsave(disp_result + image_name + '.jpg', filtered_disp, cmap = 'magma')
    count += 1
    # os.replace(image_to_disp + image_name + '.jpg', disp_result + image_name + '.jpg')







path = r'C:/Users/Hp/Desktop/monodepth2/data_from_lidar/test/'
imgL = cv2.imread(path + 'stereo_front_left_rect_315969338566583992.jpg', 0)
imgR = cv2.imread(path + 'stereo_front_right_rect_315969338566584664.jpg', 0)


block_size = 11
min_disp = -128
max_disp = 128

num_disp = max_disp - min_disp

uniquenessRatio = 5

speckleWindowSize = 200

speckleRange = 2
disp12MaxDiff = 0

stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=block_size,
    uniquenessRatio=uniquenessRatio,
    speckleWindowSize=speckleWindowSize,
    speckleRange=speckleRange,
    disp12MaxDiff=disp12MaxDiff,
    P1=8 * 1 * block_size * block_size,
    P2=32 * 1 * block_size * block_size,
)
disparity_SGBM = stereo.compute(imgL, imgR)

# Normalize the values to a range from 0..255 for a grayscale image
disparity_SGBM = cv2.normalize(disparity_SGBM, disparity_SGBM, alpha=255,
                              beta=0, norm_type=cv2.NORM_MINMAX)
disparity_SGBM = np.uint8(disparity_SGBM)
# Можно запускать карту глубины без фильтра по disparity_SGBM

wsize=31
max_disp = 128
sigma = 1.5 #Размытость default = 1.5 (>1)
lmbda = 8000.0
# left_matcher = cv2.StereoBM_create(max_disp, wsize)
left_matcher = stereo
right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
left_disp = left_matcher.compute(imgL, imgR)
right_disp = right_matcher.compute(imgR,imgL)

# Now create DisparityWLSFilter
wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)
filtered_disp = wls_filter.filter(left_disp, imgL, disparity_map_right=right_disp)

plt.imshow(filtered_disp, cmap='magma')
plt.colorbar()
plt.show()
