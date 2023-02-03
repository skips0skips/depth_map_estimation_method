import os
import pathlib
from pathlib import Path
import numpy as np
import cv2

#Получаем строку, содержащую путь к рабочей директории:
dir_path = pathlib.Path.cwd()
# Объединяем полученную строку с недостающими частями пути
data_lidar = Path(dir_path, 'data lidar')
data_model = Path(dir_path, 'data model')

file_counter = 0 #количество изображений
mae_metric = [] #список значений метрики МАЕ
coef_of_change = [] #список значений коэффициента изменения

for file in os.listdir(data_model): #возвращает список изображений в директориях
    lidar_img = cv2.imread(str(data_lidar.joinpath(file))) #считывает изображение
    neuron_img = cv2.imread(str(data_model.joinpath(file)))
    # neuron_v2 = cv2.imread(data_model + file)
    imgWidth, imgHeight, _ = lidar_img.shape #размеры изображения по вертикали и горизонтали

    mae_subtraction_modulo = []
    coef_of_change_difference = []
    mae_count_N = 0

    for x in range(imgWidth): #проходимся циклом по пикселям изображения 
        for y in range(imgHeight):
            if (lidar_img[x,y][0] != 0): #отбрасываем те пиксели лидара где нет информации об расстоянии
                coef_of_change_difference.append(neuron_img[x,y][0] / lidar_img[x,y][0])
                # print(f'модель: {neuron_img[x,y][0]} лидар: {lidar_img[x,y][0]}')
                mae_subtraction_modulo.append(abs(int(neuron_img[x,y][0]) - int(lidar_img[x,y][0])))
                mae_count_N += 1
                
    mae_metric.append(np.sum(mae_subtraction_modulo) / (mae_count_N))
    coef_of_change_ratio = np.nansum([coef_of_change_difference])
    rate = coef_of_change_ratio / (imgHeight*imgWidth)     #Коэф. изменения изображения
    coef_of_change.append(rate)

    for x in range(imgWidth):   #домножение на коэффициент изменения изображения
        for y in range(imgHeight):
            if (lidar_img[x,y][0] != 0):
                neuron_img[x,y] = rate * neuron_img[x,y]
            else: neuron_img[x,y] = 0 * neuron_img[x,y]

    difference_v1 = cv2.subtract(neuron_img, lidar_img) #вычитаем  изображения
    Conv_hsv_Gray = cv2.cvtColor(difference_v1, cv2.COLOR_BGR2GRAY) #преобразования изображения из цветового пространства в серое
    #Если значение пикселя меньше порогового значения, оно устанавливается равным 0, в противном случае устанавливается максимальное значение.
    ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU) 
    difference_v1[mask != 255] = [0, 0, 255]
    lidar_img[mask != 255] = [0, 0, 255]
    # cv2.imshow("difference_v1", difference_v1)

    together_folder = Path(dir_path, 'results','toghether',file) #путь к изображению с маской
    cv2.imwrite(str(together_folder), difference_v1) #создание файла
    file_counter += 1
info = Path(dir_path, 'results','information',"info.txt")
# info = open(str(info) + "info.txt", "w+")
with open(info, 'w+') as f:
    for i in range(len(mae_metric)):
        print('MAE метрика карты ' +  '%06d'% i +'_10_disp ' + 'равна ' + str(mae_metric[i]), file=f)
        #info.write(f"значение метрики MAE изображения  +  '%06d'% i +'_10_disp ' + 'равна ' + str(mae_metric[i]))
        print('\n', file=f)
        print('Коэффициент изменения ' +  '%06d'% i +'_10_disp ' + 'равен ' + str(coef_of_change[i]), file=f)
        print('\n', file=f)
        print('\n', file=f)






# for i in range(len(mae_metric)):
#     info.write('MAE метрика карты ' +  '%06d'% i +'_10_disp ' + 'равна ' + str(mae_metric[i]))
#     #info.write(f"значение метрики MAE изображения  +  '%06d'% i +'_10_disp ' + 'равна ' + str(mae_metric[i]))
#     info.write('\n')
#     info.write('Коэффициент изменения ' +  '%06d'% i +'_10_disp ' + 'равен ' + str(coef_of_change[i]))
#     info.write('\n')
#     info.write('\n')




#cv2.imshow("neuron_img", neuron_img)
print('конец')
cv2.waitKey(0)
