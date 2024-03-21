import eel
 # подключаем библиотеку машинного зрения
import cv2
# библиотека для вызова системных функций
import os
import subprocess as sp
# библиотека для подключения к БД
import sqlite3 as sl
# для обучения нейросетей
import numpy as np
# встроенная библиотека для работы с изображениями
from PIL import Image 


@eel.expose
def getData(id,namee):
    # получаем путь к этому скрипту
    path = os.path.dirname(os.path.abspath(__file__))
    # указываем, что мы будем искать лица по примитивам Хаара
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    # счётчик изображений
    i=0
    # расстояния от распознанного лица до рамки
    offset=50
    # получаем номер пользователя
    name = id
    secondName = namee
    # получаем доступ к камере
    video=cv2.VideoCapture(0)
    # записываем данные в БД
    con = sl.connect('people.db')
    sql = 'INSERT INTO people (id, name) values(?, ?)'
    data = [
        (name, secondName)
    ]
    with con:
        con.executemany(sql, data)

    # запускаем цикл
    while True:
        # берём видеопоток
        ret, im =video.read()
        # переводим всё в ч/б для простоты
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        # настраиваем параметры распознавания и получаем лицо с камеры
        faces=detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))
        # обрабатываем лица
        for(x,y,w,h) in faces:
            # увеличиваем счётчик кадров
            i=i+1
            # записываем файл на диск
            cv2.imwrite("dataSet/face-"+ name +'.'+ str(i) + ".jpg", gray[y-offset:y+h+offset,x-offset:x+w+offset])
            # формируем размеры окна для вывода лица
            cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)
            # показываем очередной кадр, который мы запомнили
            cv2.imshow('im',im[y-offset:y+h+offset,x-offset:x+w+offset])
            # делаем паузу
            cv2.waitKey(100)
        # если у нас хватает кадров
        if i>30:
            # освобождаем камеру
            video.release()
            # удалаяем все созданные окна
            cv2.destroyAllWindows()
            # останавливаем цикл
            break
    # получаем путь к этому скрипту
    path = os.path.dirname(os.path.abspath(__file__))
    # создаём новый распознаватель лиц
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # указываем, что мы будем искать лица по примитивам Хаара
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    # путь к датасету с фотографиями пользователей
    dataPath = path+r'/dataSet'


    # получаем картинки и подписи из датасета
    def get_images_and_labels(datapath):
        # получаем путь к картинкам
        image_paths = [os.path.join(datapath, f) for f in os.listdir(datapath)]
        # списки картинок и подписей на старте пустые
        images = []
        labels = []
        # перебираем все картинки в датасете 
        for image_path in image_paths:
            # читаем картинку и сразу переводим в ч/б
            image_pil = Image.open(image_path).convert('L')
            # переводим картинку в numpy-массив
            image = np.array(image_pil, 'uint8')
            # получаем id пользователя из имени файла
            nbr = int(os.path.split(image_path)[1].split(".")[0].replace("face-", ""))
            # определяем лицо на картинке
            faces = faceCascade.detectMultiScale(image)
            # если лицо найдено
            for (x, y, w, h) in faces:
                # добавляем его к списку картинок 
                images.append(image[y: y + h, x: x + w])
                # добавляем id пользователя в список подписей
                labels.append(nbr)
                # выводим текущую картинку на экран
                cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
                # делаем паузу
                cv2.waitKey(100)
        # возвращаем список картинок и подписей
        return images, labels

    # получаем список картинок и подписей
    images, labels = get_images_and_labels(dataPath)
    # обучаем модель распознавания на наших картинках и учим сопоставлять её лица и подписи к ним
    recognizer.train(images, np.array(labels))
    # сохраняем модель
    recognizer.save(path+r'/trainer/trainer.yml')
    # удаляем из памяти все созданные окна
    cv2.destroyAllWindows()




@eel.expose
def Detected():
    #os.startfile(task)
    # получаем путь к этому скрипту
    path = os.path.dirname(os.path.abspath(__file__))
    # создаём новый распознаватель лиц
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # добавляем в него модель, которую мы обучили на прошлых этапах
    recognizer.read(path+r'/trainer/trainer.yml')
    # указываем, что мы будем искать лица по примитивам Хаара
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    # получаем доступ к камере
    cam = cv2.VideoCapture(0)
    # настраиваем шрифт для вывода подписей
    font = cv2.FONT_HERSHEY_SIMPLEX

    # запускаем цикл
    while True:
        # получаем видеопоток
        ret, im =cam.read()
        # переводим его в ч/б
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        # определяем лица на видео
        faces=faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
        # перебираем все найденные лица
        for(x,y,w,h) in faces:
            # получаем id пользователя
            nbr_predicted,coord = recognizer.predict(gray[y:y+h,x:x+w])
            # рисуем прямоугольник вокруг лица
            cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)
            # если мы знаем id пользователя
            con = sl.connect('people.db')
            with con:
                data1 = con.execute('SELECT name FROM people WHERE `id` = ' + str(nbr_predicted))
                for row in data1:
                    nbr_predicted = row[0]
            # добавляем текст к рамке
            cv2.putText(im,str(nbr_predicted), (x,y+h),font, 1.1, (0,0,255))
            # выводим окно с изображением с камеры
            cv2.imshow('Face recognition',im)
            # делаем паузу
            cv2.waitKey(10)        
eel.init("web")
eel.start("index.html", size=(700,700))