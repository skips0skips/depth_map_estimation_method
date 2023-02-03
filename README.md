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
