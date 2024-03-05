from matplotlib import pyplot as plt


def create_graph(x, y_standart, x_valid, y_pred):
    """
    Функция создает точечный графикс с фактическими значениями и линию регрессии?.

    Parameters
    ----------
    x : ndarray
        Этот параметр содержит данные
    y_standart:
        Этот параметр содержит стандартизированные данные
    x_valid :
        Этот параметр содержит валидационные данные
    y_pred:
        Этот параметр содержит предсказанные данные
    Returns
    -------
    NoneType
    """

    # Параметры размера точек и графика
    point_size = 10
    figure_size = (8, 6)
    # Установка размеров графика
    plt.figure(figsize=figure_size)

    # Зависимость целевого столбца от конкретного столбца x1 (точечный график)
    plt.scatter(x, y_standart, s=point_size)

    # Линейная регрессия
    plt.plot(x_valid, y_pred, "r")

    # Дополнительные параметры графика
    plt.xlabel('x')
    plt.ylabel('Target')
    plt.title('Сравнение предсказанных с истинными')
    plt.legend(['Истинные значения', 'Линия регрессии'])
    plt.show()
  