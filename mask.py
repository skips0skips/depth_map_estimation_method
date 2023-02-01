from PIL import Image, ImageDraw
import os
import numpy as np
import cv2
# from matplotlib import pyplot as plt
import numpy as np
# import imageio.v2 as imageio
from collections import Counter

lidar_folder = r'C:/Users/Hp/Desktop/monodepth2/assets/best_lidar/'
# neuron_folder =r'C:/Users/Hp/Desktop/monodepth2/assets/result/'
neuron_folder =r'C:/Users/Hp/Desktop/monodepth2/assets/kirill/data/'
# neuron_save = r'C:/Users/Hp/Desktop/monodepth2/assets/to_result/neuron/'
neuron_save = r'C:/Users/Hp/Desktop/monodepth2/assets/kirill/neuron/'
# together_folder = r'C:/Users/Hp/Desktop/monodepth2/assets/to_result/toghether/'
together_folder = r'C:/Users/Hp/Desktop/monodepth2/assets/kirill/toghether/'
# info = r'C:/Users/Hp/Desktop/monodepth2/assets/to_result/info/'
info = r'C:/Users/Hp/Desktop/monodepth2/assets/kirill/info/'

# lidar_file = '55.png'
# neuron_file = '000055_10_disp.jpeg'
file_counter = 0
met = []
rating = []

for file in os.listdir(neuron_folder):
    lidar_img = cv2.imread(lidar_folder + str(file_counter) + '.png')
    neuron_img = cv2.imread(neuron_folder + file)
    # lidar_img = cv2.imread(lidar_folder + lidar_file)
    # neuron_img = cv2.imread(neuron_folder + neuron_file)
    neuron_v2 = cv2.imread(neuron_folder + file)

    imgWidth, imgHeight, _ = lidar_img.shape

    mae = []
    difference = []
    mae_count = 0

    for x in range(imgWidth):
        for y in range(imgHeight):
            if (lidar_img[x,y][0] != 0):
                diff = (neuron_img[x,y][0] / lidar_img[x,y][0])
                difference.append(diff)
                mae.append(abs(neuron_img[x,y][0] - lidar_img[x,y][0]))
                mae_count += 1

    sum = np.sum(mae)
    metric = sum / (mae_count)
    met.append(metric)
    # print(metric)     #Показать метрику МАЕ

    ratio = np.nansum([difference])
    rate = ratio / (imgHeight*imgWidth)     #Коэф. изменения изображения
    rating.append(rate)

    for x in range(imgWidth):
        for y in range(imgHeight):
            if (lidar_img[x,y][0] != 0):
                neuron_img[x,y] = rate * neuron_img[x,y]
            else: neuron_img[x,y] = 0 * neuron_img[x,y]

    for x in range(imgWidth):
        for y in range(imgHeight):
            if (lidar_img[x,y][0] == 0):
                neuron_v2[x,y] = 0 * neuron_v2[x,y]


    difference_v1 = cv2.subtract(neuron_img, lidar_img)
    Conv_hsv_Gray = cv2.cvtColor(difference_v1, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
    difference_v1[mask != 255] = [0, 0, 255]
    # neuron_img[mask != 255] = [0, 0, 255]  #2 разные маски (не очень)
    # lidar_img[mask != 255] = [0, 0, 255]
    # cv2.imshow("difference_v1", difference_v1)

    difference_v2 = cv2.subtract(neuron_v2, lidar_img)
    Conv_hsv_Gray = cv2.cvtColor(difference_v2, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
    difference_v2[mask != 255] = [0, 0, 255]
    # cv2.imshow("difference_v2", difference_v2)


    # together = cv2.addWeighted(neuron_img, 1, lidar_img, 0.2, 0.0)

    cv2.imwrite(together_folder + '/v1/' + file, difference_v1)
    cv2.imwrite(together_folder + '/v2/' + file, difference_v2)

    # cv2.imwrite(neuron_save + file, neuron_img)
    # cv2.imwrite(together_folder + file, together)
    file_counter += 1

info = open(info + "info.txt", "w+")
for i in range(len(met)):
    info.write('MAE метрика карты ' +  '%06d'% i +'_10_disp ' + 'равна ' + str(met[i]))
    info.write('\n')
    info.write('Коэффициент изменения ' +  '%06d'% i +'_10_disp ' + 'равен ' + str(rating[i]))
    info.write('\n')
    info.write('\n')




cv2.imshow("neuron_img", neuron_img)
# cv2.imshow("lidar_img", lidar_img)
# cv2.imshow("together", together)
cv2.waitKey(0)
