# подключаем библиотеку для работы с камерой и тривиальной обработки изображений
import cv2
# подключаем библиотеку для распознавания лиц и определения координат ключевых точек лица
import dlib
# подключаем библиотеку для воспроизведения звука
from playsound import playsound
# подключаем библиотеку для создания графиков
import matplotlib.pyplot as plt

import time
import os
from collections import deque


# задаем камеру
cap = cv2.VideoCapture(0)
# задаем цвет для линии маркироки глаз
color = (0, 255, 0)
# задаем толщину этих линий
thickness = 2

# создаем экземпляр детектора лица
detector = dlib.get_frontal_face_detector()
# создаем определитель ключевых точек лица, загружаем веса
predictor = dlib.shape_predictor("./data/shape_predictor_68_face_landmarks.dat")
# ключевые точки, ограничивающие левый глаз на изображении
# по часовой стрелке, от левой крайней точки
left_eye_points = [36, 37, 38, 39, 40, 41]
# и для правого глаза -//-
right_eye_points = [42, 43, 44, 45, 46, 47]

# функция определения координат середины отрезка
def middle(point1 , point2):
    middle_x = (point1.x + point2.x) / 2
    middle_y = (point1.y + point2.y) / 2
    return middle_x, middle_y

# функция определения расстояния между веками
# facial_landmarks - объект, содержащий 68 ключевых точек лица
def get_eye_lid_distance(eye_points, facial_landmarks):

    # определяем среднюю верхню, среднюю нижнюю, левую и правую точки глаза
    x_top, y_top = middle(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    x_bottom, y_bottom = middle(facial_landmarks.part(eye_points[4]), facial_landmarks.part(eye_points[5]))
    left_point = facial_landmarks.part(eye_points[0])
    x_left, y_left = left_point.x, left_point.y
    right_point = facial_landmarks.part(eye_points[3])
    x_right, y_right = right_point.x, right_point.y

    # рисуем перекрестия на глазах
    cv2.line(frame, (int(x_left), int(y_left)),
             (int(x_right), int(y_right)), color, thickness)
    cv2.line(frame, (int(x_bottom), int(y_bottom)),
             (int(x_top), int(y_top)), color, thickness)

    # определяем расстояние между крайней правой и крайней левой точками глаза в пикселях
    eye_width = ( (x_left - x_right) ** 2 + (y_left - y_right) ** 2) ** 0.5
    # определяем расстояние между веками в пикселях
    distance_between_lids = ( (x_top - x_bottom) ** 2 + (y_top - y_bottom) ** 2) ** 0.5

    # определяем относительное расстояние между веками,
    # чтобы обнаруживать моргания независимо от расстояния между пользователем и камерой         
    relative_distance = distance_between_lids / eye_width
    return relative_distance

# функция для обнаружения моргания
# frame - изображение с камеры
def wink_detection(frame):
    # переводим изображение в оттенки серого для передачи изображения детектору и предиктору
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # определяем, границы лица на изображении
    faces = detector(gray)
    average_lids_dist = float("inf")

    # проверяем, есть ли лица на фото
    if len(faces) == 0:
        # если нет, то
        print("Лицо не найдено")
    else:
        # если да, то
        face = faces[0]
        # определяем ключевые точки лица с помощью
        # алгоритма определения ключевых точек с учетом ориентации лица на изображении
        landmarks = predictor(gray, face)
        # определяем расстояния между веками правого и левого глаза
        left_lids_dist = get_eye_lid_distance(left_eye_points, landmarks)
        right_lids_dist = get_eye_lid_distance(right_eye_points, landmarks)
        # считаем среднее арифметическое двух этих расстояний, чтобы потом могли отличить
        # моргание двумя глазами от подмигивания одним глазом
        average_lids_dist = (left_lids_dist + right_lids_dist) / 2 
        
    # 0.21 - состояние при котором веки закрыты и произошло моргание
    # выясненное эмпирическим путем, чувствительность детектора
    # чем ниже значение, тем выше чувствительность
    return average_lids_dist < 0.21

# функция обновления состояния глаз
def wink_speed_update():
    global previous_moment, wink_counter, average_speed, calibration_ended, speed
    # определяем время между текущим моментом и моментом последнего моргания
    delta = time.time() - previous_moment
    # вычисляем частоту моргания на текущий момент
    speed = int((wink_counter / delta) * 60)

    # если прошло время калибровки, высчитываем среднюю частоту моргания
    if (calibration_period - 1) <= delta <= (calibration_period + 1):
        if not calibration_ended:
            average_speed = int((wink_counter / delta) * 60 )
            calibration_ended = True

    # очищаем окно терминала
    os.system("cls")

    if calibration_ended:
        # каждые 100 кадров обнуляем счетчик морганий и предыдущее время замера
        if frame_counter == 0:
            wink_counter = 0
            previous_moment = time.time()
        print(f"Средняя частота моргания: {average_speed} морганий в минуту")
    else:
        # если еще не прошло время калибровки
        print("Идет подсчет средней частоты моргания...")
        
    # если скорость снизилась
    if speed < average_speed:
        # подаем сигнал
        playsound("./data/beep.mp3", False)
        print("Внимание! Частота моргания понизилась!")
    print(f"Частота моргания сейчас: {speed} морганий в минуту")

recently_winked = False
previous_moment, absolute_start = time.time(), time.time()
wink_counter, frame_counter = 0, 0
average_speed, speed = 0, 0
calibration_period = 3 * 60 # калибровка 3 минуты
calibration_ended = False
plot_period = 230 # длина "истории графиков" в кадрах
# очереди для более легкого обновления значений точек в графиках
time_q1, time_q2, freq_q, wink_q = deque([0]*plot_period,maxlen=plot_period), deque([0]*plot_period,maxlen=60), deque([0]*plot_period, maxlen=plot_period), deque([0]*plot_period, maxlen=60)

# создаем болванки графиков, подписываем оси
fig, (freq_plot, wink_plot) = plt.subplots(2, 1, figsize = (6, 12))
freq_plot.set_title("Частота моргания")
freq_plot.set_ylabel("Средняя частота (моргания в минуту)")
freq_plot.set_xlabel("Время (с)")
data_freq, = freq_plot.plot([], [])
wink_plot.set_title("График фронта мигательного движения")
wink_plot.set_ylabel("Состояние век (0-закрыты, 1-открыты)")
wink_plot.set_ylim(0, 1.3)
wink_plot.set_xlabel("Время (с)")
data_wink, = wink_plot.plot([], [])

# прорисовываем окно графиков
plt.ion()
plt.show()

while True:
    _, frame = cap.read()
    # каждые 40 кадров обновляем скорость
    frame_counter = (frame_counter + 1) % 100

    # wink_detected - моргнул ли ползователь: True или False
    wink_detected = wink_detection(frame)
    
    # каждые 25 кадров обновляем показатель скорости
    if frame_counter % 25 == 0:
        wink_speed_update()

    # кладем в очереди новые значения, в это время старые, вышедшие за пределы истории
    # значения стираются
    delta_time = time.time() - absolute_start
    time_q1.append(delta_time)
    time_q2.append(delta_time)
    wink_q.append(int(wink_detected))
    freq_q.append(speed)
    # обновляем графики
    data_freq.set_data(list(time_q1), list(freq_q))
    freq_plot.relim()
    freq_plot.autoscale_view()
    data_wink.set_data(list(time_q2), list(wink_q))
    wink_plot.relim()
    wink_plot.autoscale_view(scaley=False)
    # прорисовываем их
    plt.draw()

    if wink_detected and not recently_winked:
        wink_counter += 1
        recently_winked = True
    if not wink_detected:
        recently_winked = False
    # если нажали кнопку Q, выходим и закрываем окна
    if cv2.waitKey(1) == ord("q"):
        break
    # показываем фото с отмеченными глазами
    cv2.imshow("WinkDetector", frame)

cap.release()
cv2.destroyAllWindows()
