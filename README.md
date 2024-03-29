## Инструкция по запуску программы
1. Клонируйте репозиторий на свой локальный компьютер
   
2. Запустите программу в IDE с тестовым файлом и сверьте свой график с графиком ниже.


![График на тестовых данных](https://github.com/JuliaDrama/Task2/blob/main/photo_2024-03-05_19-50-26.jpg)


   
На вход программе подается файл и по его данным строятся графики линейной регрессии
Файл состоит из двух столбцов.

### Формат данных

Первая строка это названия столбцов - time и count_messages, написанные через запятую.

Следующие строчки это данные, которые также записаны через запятую. 

Первое значение время в формате **HH:MI:SS**.

Второе значение **N.000000**, где N - количество сообщений

**Первые 8 строк тестового файла:**

time,count_messages

00:00:00,0.000000

00:00:01,1.000000

00:00:02,0.000000

00:00:03,6.000000

00:00:04,7.000000

00:00:05,0.000000

00:00:06,1.000000

## Установка зависимостей
`pip install -r requirements.txt`

Таким образом, сразу установятся все необходимые пакеты

## Описание проекта
В процессе выполнения проекта код из задания 1 был раздел на модули: *graph.py*, *regression.py* и файл *main.py*

**graph.py** содержит функции для отрисовки графиков

**regression.py** содержит функции для создания и обучения модели

**main.py** основной файл программы

### График линейной регрессии первой модели, соазданной с помощью библиотек.
![График линейной регрессии первой модели, соазданной с помощью библиотек](https://github.com/JuliaDrama/Task2/blob/main/%D0%93%D1%80%D0%B0%D1%84%D0%B8%D0%BA%20%D0%BC%D0%BE%D0%B4%D0%B5%D0%BB%D1%8C1.jpg)

### График линейной регрессии второй модели, вычисленной аналитически.
![График линейной регрессии второй модели, вычисленной аналитически](https://github.com/JuliaDrama/Task2/blob/main/%D0%93%D1%80%D0%B0%D1%84%D0%B8%D0%BA%20%D0%BC%D0%BE%D0%B4%D0%B5%D0%BB%D1%8C%202.jpg)

# Вывод
Mean Squared Error for the first model: 367.11476560214453

Mean Squared Error for the second model: 367.111792679908

Оба значения MSE (Mean Squared Error) очень близки, но вторая модель имеет немного меньшее значение MSE с некоторой погрешностью, однако это всё равно говорит о том, что обе модели плохо справляются с предсказыванием.
