
import pandas as pd
import numpy as np
import sklearn
# из модуля preprocessing библиотеки sklearn имрорт функциии StandardScaler
from sklearn.preprocessing import StandardScaler
# импорт из модуля linear_model функцию для создания модели линейной регресии
from sklearn.linear_model import LinearRegression


def process_data(dataframe: pd.DataFrame) -> tuple:
    """
    Функция переводит столбец time формата '%H:%M:%S' в секунды.

    Parameters
    ----------
    dataframe : DataFrame
        Этот параметр является объектом класса библиотеки Pandas.

    Returns
    -------
    tuple
        X : DataFrame  
            Объект класса DataFrame библиотеки Pandas.
        y : Series  
            Объект класса Series библиотеки Pandas.
    """

    x = dataframe['time']  # столбец с целевой переменной target (вектор)
    y = dataframe['count_messages']  # признак

    x = pd.to_datetime(dataframe['time'], format='%H:%M:%S')
    x = ((x.dt.hour * 60+x.dt.minute)
         * 60 + x.dt.second).to_frame()

    return x, y


def make_standard(x: pd.DataFrame) -> np.ndarray:
    """
    Функция стандартизирует данные.

    Parameters
    ----------
    x : DataFrame
        Этот параметр является объектом класса библиотеки Pandas.

    Returns
    -------
    x : ndarray  
        Стандартизированные данные, представленные в виде массива NumPy.
    """
    sc = StandardScaler()  # Создание объекта стандартизатора

    # Применение стандартизации к признакам X и сохранение результата в X_c
    x_c = sc.fit_transform(x)
    return x_c


def create_model(x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray) -> np.ndarray:
    """
    Функция создает модель, обучает её и возвращает предсказанные данные

    Parameters
    ----------
    x_train : ndarray
        Массив NumPy, содержащий тренировочные данные.
    y_train: Series
        Массив NumPy, содержащий целевые значения для тренировочных данных.
    x_valid: ndarray
         Массив NumPy, содержащий валидационные данные.

    Returns
    -------
    y_predicted : ndarray  
        Предсказанные данные, представленные в виде массива NumPy
    """
    model = LinearRegression()  # инициализирование модели LinearRegressionS
    model.fit(x_train, y_train)  # обучение модели на тренировочной выборке
    # предсказания модели на валидационной выборке
    y_predicted = model.predict(np.array(x_valid).reshape(-1, 1))
    return y_predicted


def compute_model(x_standart: pd.DataFrame, y: pd.Series, x_valid):
    """
    Функция вычисляет модель методом наименьших квадратов, обучает её и возвращает предсказанные данные.

    Parameters
    ----------
    x_standart : DataFrame
        Этот параметр содержит данные для вычисления модели.
    y : Series
        Этот параметр содержит целевые значения для данных.
    x_valid: ndarray
        Этот параметр содержит валидационные данные.

    Returns
    -------
    y_predicted : ndarray  
        Предсказанные данные, представленные в виде массива NumPy
    """
    y_array = np.array(y)  # Приведение в тип данных массив
    x_mean = np.mean(x_standart)
    y_mean = np.mean(y_array)
    sqrt_x_array = np.array([x_i ** 2 for x_i in x_standart])
    mean_sqrt_x = np.mean(sqrt_x_array)
    prod_x_y = np.array([x_i * y_i for x_i, y_i in zip(x_standart, y_array)])
    mean_prod_x_y = np.mean(prod_x_y)
    b = (y_mean * x_mean - mean_prod_x_y) / (x_mean ** 2 - mean_sqrt_x)
    a = y_mean - b * x_mean
    y_predicted = np.array([a + b * x for x in x_valid])
    return y_predicted
