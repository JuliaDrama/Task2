import pandas as pd
import numpy as np
import regression as rg
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error  # импорт метрик для оценки данных
import graph


def main():
    df = pd.read_csv('example_data.txt', sep=',')

    x, y = rg.process_data(df)




    y_c = rg.make_standard(y)
    print(y_c)

    print([x_i for x_i in x])
    # Разделение выборки на тренировочный и валидационный наборы с
    # помощью train_test_split, X- признак, y – целевой признак,

    x_training, x_validation, y_training, y_validation = train_test_split(x,
                                                                        y_c,
                                                                        test_size=0.25,
                                                                        random_state=0)
    y_predicted_model1 = rg.create_model(
        x_train=x_training, y_train=y_training, x_valid=x_validation)

    # датафрейм для сравнения предсказанных значений и реальных
    df_match = pd.DataFrame(
        {'Actual': y_validation, 'Predicted': y_predicted_model1})
    print(df_match)

    df_match = df_match.reset_index(drop=True)  # сброс индексации

    print(df_match)

    print('Mean Squared Error for the first model:', mean_squared_error(y_validation,
        y_predicted_model1))  # Расчет среднеквадратической ошибки (MSE) для первой модели


    graph.create_graph(x=x, y_standart=y_c, x_valid=x_validation,
                    y_pred=y_predicted_model1)

    df['time'] = x['time']
    df['count_messages']=y_c

    y_predicted_model2 = rg.compute_model(
        df=df, x_valid=x_validation)

    print('Mean Squared Error for the second model:', mean_squared_error(y_validation,
                                                                        y_predicted_model2))  # Расчет среднеквадратической ошибки (MSE) для второй модели


    graph.create_graph(x=x, y_standart=y_c, x_valid=x_validation,
                    y_pred=y_predicted_model2)


if __name__ == "__main__":
    main()

