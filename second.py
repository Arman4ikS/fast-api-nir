import numpy as np  # Импорт библиотеки NumPy для работы с массивами
import pandas as pd  # Импорт библиотеки Pandas для работы с данными в виде таблиц
import os  # Импорт модуля os для взаимодействия с операционной системой
import matplotlib.pyplot as plt  # Импорт библиотеки Matplotlib для визуализации данных
import seaborn as sns  # Импорт библиотеки Seaborn для создания более стильных графиков
from sklearn.ensemble import RandomForestRegressor  # Импорт модели RandomForestRegressor из библиотеки scikit-learn
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error  # Импорт метрик для оценки модели
from sklearn.linear_model import LinearRegression  # Импорт линейной регрессии из scikit-learn
from sklearn.preprocessing import StandardScaler  # Импорт стандартизатора данных из scikit-learn
from pylab import rcParams  # Импорт настроек рисования графиков
import math  # Импорт модуля math для математических операций
#import xgboost  # Импорт библиотеки XGBoost для градиентного бустинга
import time  # Импорт модуля time для работы со временем
#from tqdm import tqdm  # Импорт библиотеки tqdm для отображения прогресса выполнения операций
#import keras.models  # Импорт модуля keras.models для работы с нейронными сетями
#import keras.layers  # Импорт модуля keras.layers для создания слоев нейронных сетей
from sklearn.model_selection import train_test_split  # Импорт функции для разделения данных на обучающую и тестовую выборки
import tensorflow as tf  # Импорт библиотеки TensorFlow для создания и обучения нейронных сетей
from keras.models import Sequential  # Импорт модели Sequential из TensorFlow/Keras
from tensorflow.keras import layers  # Импорт модуля layers из TensorFlow/Keras для создания слоев нейронных сетей
import warnings  # Импорт модуля warnings для управления предупреждениями
import pickle

# Отключение предупреждений
warnings.simplefilter('ignore')

# Загрузка данных обучения, тестирования и RUL из файлов
data_train = pd.read_csv("train_FD001.txt", sep=" ", header=None)
data_test = pd.read_csv("test_FD001.txt", sep=" ", header=None)
data_RUL = pd.read_csv("RUL_FD001.txt", sep=" ", header=None)

# Копирование данных для последующей обработки
train_copy = data_train
test_copy = data_test

# Вывод данных обучения
data_train

# Вывод данных тестирования
data_test

# Вывод данных RUL
data_RUL

# Удаление последних двух столбцов из данных обучения
data_train.drop(columns=[26, 27], inplace=True)

# Удаление последних двух столбцов из данных тестирования
data_test.drop(columns=[26, 27], inplace=True)

# Удаление второго столбца из данных RUL
data_RUL.drop(columns=[1], inplace=True)

# Имена столбцов для данных обучения
columns_train = ['unit_ID', 'cycles', 'setting_1', 'setting_2', 'setting_3', 'T2', 'T24', 'T30', 'T50', 'P2', 'P15',
                 'P30', 'Nf',
                 'Nc', 'epr', 'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'farB', 'htBleed', 'Nf_dmd', 'PCNfR_dmd', 'W31',
                 'W32']

# Присвоение имен столбцам данных обучения
data_train.columns = columns_train

# Вывод описательных статистик данных обучения
data_train.describe()

# Функция для добавления RUL в данные


def add_rul(g):
    g['RUL'] = max(g['cycles']) - g['cycles']  # Расчет RUL
    return g


# Применение функции add_rul к данным обучения, сгруппированным по unit_ID
train = data_train.groupby('unit_ID').apply(add_rul)

# Вывод первых строк обработанных данных
train.head()

# Группировка данных обучения по unit_ID и подсчет максимального количества циклов для каждого двигателя
cnt_train = train[["unit_ID", "cycles"]].groupby("unit_ID").max().sort_values(by="cycles", ascending=False)

# Список индексов
cnt_ind = [str(i) for i in cnt_train.index.to_list()]

# Список значений количества циклов для каждого двигателя
cnt_val = list(cnt_train.cycles.values)

# Создание графика, отображающего количество циклов для каждого двигателя
plt.style.use("seaborn")
plt.figure(figsize=(12, 30))
sns.barplot(x=list(cnt_val), y=list(cnt_ind), palette='magma')
plt.xlabel('Number of Cycles')
plt.ylabel('Engine Id')
plt.title('Number of Cycles for Engines', fontweight='bold', fontsize=24, pad=15)

# Отображение графика
plt.show()

# Создание графика, отображающего стандартное отклонение каждого столбца данных обучения
plt.figure(figsize=(18, 9))
subset_stats = data_train.agg(['mean', 'std']).T[2:]
ax = sns.barplot(x=subset_stats.index, y="std", data=subset_stats, palette='magma')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.set_xlabel("Sensor")
ax.set_ylabel("Standard Deviation")
ax.set_title("Standard Deviation of Each Column", fontweight='bold', fontsize=24, pad=15)

# Добавление значений стандартного отклонения над столбцами
for p in ax.patches:
    ax.annotate(str(round(p.get_height(), 3)), (p.get_x() * 1.005, p.get_height() * 1.005))

# Отображение графика
plt.show()

# Удаление ненужных столбцов из данных обучения
train.drop(columns=['Nf_dmd', 'PCNfR_dmd', 'P2', 'T2', 'setting_3', 'farB', 'epr'], inplace=True)

# Создание тепловой карты корреляции между признаками данных обучения
sns.heatmap(train.corr(), annot=True, cmap='Blues', linewidths=0.2)
fig = plt.gcf()
fig.set_size_inches(20, 20)

# Отображение тепловой карты
plt.show()


# Определение функции для расчета RUL
def process_targets(data_length, early_rul=None):
    if early_rul == None:
        return np.arange(data_length - 1, -1, -1)
    else:
        early_rul_duration = data_length - early_rul
        if early_rul_duration <= 0:
            return np.arange(data_length - 1, -1, -1)
        else:
            return np.append(early_rul * np.ones(shape=(early_rul_duration,)), np.arange(early_rul - 1, -1, -1))


# Определение функции для обработки входных данных с целями
def process_input_data_with_targets(input_data, target_data=None, window_length=1, shift=1):
    num_batches = np.int(np.floor((len(input_data) - window_length) / shift)) + 1
    num_features = input_data.shape[1]
    output_data = np.repeat(np.nan, repeats=num_batches * window_length * num_features).reshape(num_batches,
                                                                                                window_length,
                                                                                                num_features)
    if target_data is None:
        for batch in range(num_batches):
            output_data[batch, :, :] = input_data[(0 + shift * batch):(0 + shift * batch + window_length), :]
        return output_data
    else:
        output_targets = np.repeat(np.nan, repeats=num_batches)
        for batch in range(num_batches):
            output_data[batch, :, :] = input_data[(0 + shift * batch):(0 + shift * batch + window_length), :]
            output_targets[batch] = target_data[(shift * batch + (window_length - 1))]
        return output_data, output_targets


# Определение функции для обработки тестовых данных
def process_test_data(test_data_for_an_engine, window_length, shift, num_test_windows=1):
    max_num_test_batches = np.int(np.floor((len(test_data_for_an_engine) - window_length) / shift)) + 1
    if max_num_test_batches < num_test_windows:
        required_len = (max_num_test_batches - 1) * shift + window_length
        batched_test_data_for_an_engine = process_input_data_with_targets(test_data_for_an_engine[-required_len:, :],
                                                                          target_data=None,
                                                                          window_length=window_length, shift=shift)
        return batched_test_data_for_an_engine, max_num_test_batches
    else:
        required_len = (num_test_windows - 1) * shift + window_length
        batched_test_data_for_an_engine = process_input_data_with_targets(test_data_for_an_engine[-required_len:, :],
                                                                          target_data=None,
                                                                          window_length=window_length, shift=shift)
        return batched_test_data_for_an_engine, num_test_windows


# Загрузка тестовых данных из файла
test_data = pd.read_csv("test_FD001.txt", sep=" ", header=None, names=columns_train)

# Загрузка фактического RUL из файла
true_rul = pd.read_csv("RUL_FD001.txt", sep=' ', header=None)

# Установка параметров для обработки данных
window_length = 30
shift = 1
early_rul = 125
processed_train_data = []
processed_train_targets = []
num_test_windows = 5
processed_test_data = []
num_test_windows_list = []

# Список столбцов, которые необходимо удалить
columns_to_be_dropped = ['unit_ID', 'setting_1', 'setting_2', 'setting_3', 'T2', 'P2', 'P15', 'P30', 'epr',
                         'farB', 'Nf_dmd', 'PCNfR_dmd']

# Сохранение первого столбца данных обучения
train_data_first_column = data_train["unit_ID"]

# Сохранение первого столбца тестовых данных
test_data_first_column = test_data["unit_ID"]

# Стандартизация данных обучения
scaler = StandardScaler()
train_data = scaler.fit_transform(data_train.drop(columns=columns_to_be_dropped))

# Стандартизация тестовых данных
test_data = scaler.transform(test_data.drop(columns=columns_to_be_dropped))

# Преобразование массивов в датафреймы
train_data = pd.DataFrame(data=np.c_[train_data_first_column, train_data])
test_data = pd.DataFrame(data=np.c_[test_data_first_column, test_data])

# Получение количества уникальных двигателей в данных обучения и тестовых данных
num_train_machines = len(train_data[0].unique())
num_test_machines = len(test_data[0].unique())

# Итерация по каждому уникальному двигателю в данных обучения
for i in np.arange(1, num_train_machines + 1):
    # Получение данных обучения для текущего двигателя
    temp_train_data = train_data[train_data[0] == i].drop(columns=[0]).values

    # Проверка наличия достаточного количества данных для заданной длины окна
    if len(temp_train_data) < window_length:
        print("Train engine {} doesn't have enough data for window_length of {}".format(i, window_length))
        raise AssertionError("Window length is larger than number of data points for some engines. "
                             "Try decreasing window length.")

    # Создание массива целей для текущего двигателя
    temp_train_targets = process_targets(data_length=temp_train_data.shape[0], early_rul=early_rul)

    # Обработка данных и целей для текущего двигателя
    data_for_a_machine, targets_for_a_machine = process_input_data_with_targets(temp_train_data, temp_train_targets,
                                                                                window_length=window_length,
                                                                                shift=shift)

    # Добавление обработанных данных и целей в соответствующие списки
    processed_train_data.append(data_for_a_machine)
    processed_train_targets.append(targets_for_a_machine)

# Объединение обработанных данных и целей
processed_train_data = np.concatenate(processed_train_data)
processed_train_targets = np.concatenate(processed_train_targets)

# Итерация по каждому уникальному двигателю в тестовых данных
for i in np.arange(1, num_test_machines + 1):
    # Получение тестовых данных для текущего двигателя
    temp_test_data = test_data[test_data[0] == i].drop(columns=[0]).values

    # Проверка наличия достаточного количества данных для заданной длины окна
    if len(temp_test_data) < window_length:
        print("Test engine {} doesn't have enough data for window_length of {}".format(i, window_length))
        raise AssertionError("Window length is larger than number of data points for some engines. "
                             "Try decreasing window length.")

    # Подготовка тестовых данных
    test_data_for_an_engine, num_windows = process_test_data(temp_test_data, window_length=window_length, shift=shift,
                                                             num_test_windows=num_test_windows)

    # Добавление обработанных тестовых данных в соответствующий список
    processed_test_data.append(test_data_for_an_engine)
    num_test_windows_list.append(num_windows)

# Объединение обработанных тестовых данных
processed_test_data = np.concatenate(processed_test_data)
true_rul = true_rul[0].values

# Перемешивание обучающих данных
index = np.random.permutation(len(processed_train_targets))
processed_train_data, processed_train_targets = processed_train_data[index], processed_train_targets[index]

# Вывод размерности обработанных данных
print("Processed trianing data shape: ", processed_train_data.shape)
print("Processed training ruls shape: ", processed_train_targets.shape)
print("Processed test data shape: ", processed_test_data.shape)
print("True RUL shape: ", true_rul.shape)

# Разделение данных обучения на обучающую и валидационную выборки
processed_train_data, processed_val_data, processed_train_targets, processed_val_targets = train_test_split(
    processed_train_data,
    processed_train_targets,
    test_size=0.2,
    random_state=83)

# Вывод размерности данных после разделения
print("Processed train data shape: ", processed_train_data.shape)
print("Processed validation data shape: ", processed_val_data.shape)
print("Processed train targets shape: ", processed_train_targets.shape)
print("Processed validation targets shape: ", processed_val_targets.shape)

'''
# Определение функции для создания скомпилированной модели
def create_compiled_model():
    model = Sequential([
        layers.LSTM(128, input_shape=(window_length, 14), return_sequences=True, activation="tanh"),
        # LSTM слой с 128 нейронами
        layers.LSTM(64, activation="tanh", return_sequences=True),  # LSTM слой с 64 нейронами
        layers.LSTM(32, activation="tanh"),  # LSTM слой с 32 нейронами
        layers.Dense(96, activation="relu"),  # Полносвязный слой с 96 нейронами и функцией активации ReLU
        layers.Dense(128, activation="relu"),  # Полносвязный слой с 128 нейронами и функцией активации ReLU
        layers.Dense(1)  # Выходной слой с одним нейроном
    ])
    model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))  # Компиляция модели
    return model


# Определение функции для изменения learning rate во время обучения
def scheduler(epoch):
    if epoch < 5:
        return 0.001
    else:
        return 0.0001


# Создание callback для изменения learning rate
callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)

# Создание и компиляция модели
model = create_compiled_model()

# Обучение модели
history = model.fit(processed_train_data, processed_train_targets, epochs=10,
                    validation_data=(processed_val_data, processed_val_targets),
                    callbacks=callback,
                    batch_size=128, verbose=2)


# Повторное определение функции для изменения learning rate
def scheduler(epoch):
    if epoch < 5:
        return 0.001
    else:
        return 0.0001


# Создание callback для изменения learning rate
callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
'''
# Создание и компиляция модели
#model = create_compiled_model()
f = open('model.pkl','r')
model = pickle.load(f)
f.close()
# Обучение модели
history = model.fit(processed_train_data, processed_train_targets, epochs=10,
                    validation_data=(processed_val_data, processed_val_targets),
                    callbacks=callback,
                    batch_size=128, verbose=2)

# Прогнозирование RUL для тестовых данных с помощью модели
rul_pred = model.predict(processed_test_data).reshape(-1)

# Разбиение предсказанных значений RUL по каждому двигателю
preds_for_each_engine = np.split(rul_pred, np.cumsum(num_test_windows_list)[:-1])

# Вычисление среднего значения RUL для каждого двигателя
mean_pred_for_each_engine = [np.average(ruls_for_each_engine, weights=np.repeat(1 / num_windows, num_windows))
                             for ruls_for_each_engine, num_windows in zip(preds_for_each_engine, num_test_windows_list)]

# Вычисление RMSE между фактическими и предсказанными значениями RUL
RMSE = np.sqrt(mean_squared_error(true_rul, mean_pred_for_each_engine))
print("RMSE: ", RMSE)

# Сохранение модели
#tf.keras.models.save_model(model, "FD001_LSTM_piecewise_RMSE_" + str(np.round(RMSE, 4)) + ".h5")

# Выборка индексов последних примеров из каждого двигателя
indices_of_last_examples = np.cumsum(num_test_windows_list) - 1

# Получение предсказанных значений RUL для последних примеров
preds_for_last_example = np.concatenate(preds_for_each_engine)[indices_of_last_examples]

# Вычисление RMSE для последних примеров
RMSE_new = np.sqrt(mean_squared_error(true_rul, preds_for_last_example))
print("RMSE (Taking only last examples): ", RMSE_new)


# Вычисление S-score для последних примеров
def compute_s_score(rul_true, rul_pred):
    diff = rul_pred - rul_true
    return np.sum(np.where(diff < 0, np.exp(-diff / 13) - 1, np.exp(diff / 10) - 1))


s_score = compute_s_score(true_rul, preds_for_last_example)
print("S-score: ", s_score)

# Визуализация фактических и предсказанных значений RUL
plt.plot(true_rul, label="True RUL", color="orange")
plt.plot(preds_for_last_example, label="Pred RUL", color="blue")
plt.legend()
plt.show()