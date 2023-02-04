# Иследование с помощью метрик количественной и качественной оценки карты глубины
Этот репозиторий содержит код для определения качества работы построения 2D карт глубины. Репозиторий сопровождает статью:....
Оценке подвергались две модели ([MiDaS](https://github.com/isl-org/MiDaS)) и ([monodepth 2](https://github.com/nianticlabs/monodepth2)) в условиях дорожно-транспортных
сцен. Для проведения анализа использовался открытый дата сет ([KITTI](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion,%202022)) с 
необработанными лидарными сканированиями и RGB-изображениями опубликованный в 
([Sparsity Invariant CNNs (THREEDV 2017)](https://www.cvlibs.net/publications/Uhrig2017THREEDV.pdf)).

В таблице представлена метрика оценки MAE. С помощью данной метрики можно скомпенсировать шум и различные артефакты в наборе данных. Коэффициент изменения, 
показывает среднее число, на которое необходимо умножить каждый пиксель, чтобы изображение, полученное с модели, было максимально близко к значениям лидара. 
| Номер изображения | Метрика МАЕ | Коэффициент изменения |
|----------------|:----------------:|:----------------:|
| 1 | 60.3935 | 0.5393 |
| 2 | 83.4511 | 0.7936 |
| 3 | 82.8349 | 0.7659 |

На рисунке ниже продемонстрирован качественный метод оценки глубины. Красным цветом обозначеныте пиксели расстояние в которых расстояние полученное с лидара равно
расстоянию, полученному с модели.

## Настройка
Настройка данных:
1.  Скачайте набор данных для тестирования ([KITTI](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion,%202022)) (Download manually selected validation and test data sets (2 GB)) либо воспользуйтесь данными из статьи ([ссылка](https://drive.google.com/file/d/1cuGYm2giIQouX8ETWb_1zSo1Y-P07QQl/view?usp=sharing)).
2.  Выберете тестируемую модель оценки карт глубины и произведите построение (либо воспользуйте готовыми изображениями которые использовались в статье ([ссылка](https://drive.google.com/file/d/1n7ojG7fpmrM3TG5GMSCHPLZ5eET0BBtx/view?usp=sharing)).
3.  Изображения с лидара помещаются в папку `data lidar`, изображения из модели вставляются в `data model`.

Настройка зависимостей:

Для работы программы необходимо установить OpenCV: `pip install opencv-python`.
## Использование
1. Запустите программу: `python run.py`.
2. По окончанию работы в командной строке должно появится сообщение: `Программа завершила работу`.
3. Результирующие данные записываются в папку `results`.
4. В данной папке есть два раздела в `information` выводится файл с метрикой МАЕ и Коэффициентом изменения по каждому изображению, а в `difference image` изображения демонстрирующие различие между тестируемой моделью и данными с лидара.
