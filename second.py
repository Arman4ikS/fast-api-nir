import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from io import BytesIO
import keras
from keras import models


def processing(testData, trueRUL):
    with open("file.txt", "w+") as f:
        f.write(testData.decode("utf-8"))
    data_test = pd.read_csv("file.txt", sep=" ", header=None)
    os.remove("file.txt")

    data_test.drop(columns=[26, 27], inplace=True)

    columns_test = ['unit_ID', 'cycles', 'setting_1', 'setting_2', 'setting_3', 'T2', 'T24', 'T30', 'T50', 'P2', 'P15',
                    'P30', 'Nf', 'Nc', 'epr', 'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'farB', 'htBleed', 'Nf_dmd',
                    'PCNfR_dmd', 'W31', 'W32']

    data_test.columns = columns_test

    columns_to_be_dropped = ['unit_ID', 'setting_1', 'setting_2', 'setting_3', 'T2', 'P2', 'P15', 'P30', 'epr',
                             'farB', 'Nf_dmd', 'PCNfR_dmd']

    scaler = StandardScaler()
    test_data = scaler.fit_transform(data_test.drop(columns=columns_to_be_dropped))

    test_data = pd.DataFrame(data=np.c_[data_test['unit_ID'], test_data])

    def process_input_data_with_targets(input_data, target_data=None, window_length=1, shift=1):
        num_batches = int(np.floor((len(input_data) - window_length) / shift)) + 1
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

    def process_test_data(test_data_for_an_engine, window_length, shift, num_test_windows=1):
        max_num_test_batches = int(np.floor((len(test_data_for_an_engine) - window_length) / shift)) + 1
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

    window_length = 30
    shift = 1
    num_test_windows = 5

    num_test_machines = len(test_data[0].unique())
    processed_test_data = []
    num_test_windows_list = []

    for i in np.arange(1, num_test_machines + 1):
        temp_test_data = test_data[test_data[0] == i].drop(columns=[0]).values
        if len(temp_test_data) < window_length:
            raise AssertionError(f"Test engine {i} doesn't have enough data for window_length of {window_length}.")
        test_data_for_an_engine, num_windows = process_test_data(temp_test_data, window_length=window_length, shift=shift,
                                                                 num_test_windows=num_test_windows)
        processed_test_data.append(test_data_for_an_engine)
        num_test_windows_list.append(num_windows)

    processed_test_data = np.concatenate(processed_test_data)

    model = keras.models.load_model("/app/model.h5")

    rul_pred = model.predict(processed_test_data).reshape(-1)

    preds_for_each_engine = np.split(rul_pred, np.cumsum(num_test_windows_list)[:-1])

    indices_of_last_examples = np.cumsum(num_test_windows_list) - 1

    preds_for_last_example = np.concatenate(preds_for_each_engine)[indices_of_last_examples]

    plt.figure(figsize=(14, 6))

    with open("file2.txt", "w+") as f:
        f.write(trueRUL.decode("utf-8"))
    trueRul = pd.read_csv("file2.txt", sep=" ", header=None)
    os.remove("file2.txt")

    plt.subplot(2, 1, 1)
    plt.plot(trueRul, label="True RUL", color="orange")
    plt.title("True RUL")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(preds_for_last_example, label="Pred RUL", color="blue")
    plt.title("Pred RUL")
    plt.legend()
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    plt.clf()

    return buffer, preds_for_last_example

