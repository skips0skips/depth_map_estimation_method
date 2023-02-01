from PIL import Image, ImageDraw
import os
from matplotlib import pyplot as plt
import numpy as np
import cv2

disp_0 = r'C:/Users/Hp/Desktop/BEST/training/disp_occ_0/'
disp_1 = r'C:/Users/Hp/Desktop/BEST/training/disp_occ_1/'
to_save = r'C:/Users/Hp/Desktop/monodepth2/assets/best_lidar/'
del_image = r'C:/Users/Hp/Desktop/monodepth2/assets/images/'

count = 0
for file in os.listdir(disp_0):
    lid = cv2.imread(disp_0 + file, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(to_save + str(count) + '.png', lid)
    count += 1

for file in os.listdir(del_image):
    file_name, file_ext = os.path.splitext(file)
    if (file_name[-2:] == '11'):
        os.remove(del_image + file)


# lid = cv2.imread(disp, cv2.IMREAD_GRAYSCALE)
# plt.imshow(lid)
# plt.show()
