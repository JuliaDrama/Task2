
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
    x['time'] =x['time'].astype (int)
    return x, y


def make_standard(y_data: pd.DataFrame) -> np.ndarray:
    """
    Функция стандартизирует данные.

    Parameters
    ----------
    y_data : DataFrame
        Этот параметр является объектом класса библиотеки Pandas.

    Returns
    -------
     : ndarray  
        Стандартизированные данные, представленные в виде массива NumPy.
    """

    # Применение стандартизации к y и сохранение результата в y_c

    y_c = (y_data - y_data.min()) / (y_data.max() - y_data.min())
    return y_c


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


def compute_model(df: pd.DataFrame, x_valid):
    """
    Функция вычисляет модель методом наименьших квадратов, обучает её и возвращает предсказанные данные.

    Parameters
    ----------
    x : DataFrame
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
    x = df['time']
    y_standart = df['count_messages']
    x_mean = np.mean(x)
    y_mean = np.mean(y_standart)
    sqrt_x = x ** 2
    mean_sqrt_x = np.mean(sqrt_x)
    prod_x_y = (x * y_standart)
    mean_prod_x_y = np.mean(prod_x_y)
    b = (y_mean * x_mean - mean_prod_x_y) / (x_mean ** 2 - mean_sqrt_x)
    a = y_mean - b * x_mean
    y_predicted = x_valid * b + a
    return y_predicted

