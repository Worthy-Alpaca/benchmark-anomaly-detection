import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from datetime import datetime
import os
import matplotlib.dates as mpl_dates
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

PACKAGE_PARENT = '../'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
DATA_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT))

DATA_PATH1 = DATA_PATH + '/data/Measurements/Measurement1/02_SPECTRUM; ch2; 1-3200Hz; m_s^2_2021_11_17_13_23_28_260.csv'

data = pd.read_csv(os.path.normpath(DATA_PATH1), names=['time', 'amplitude', 'phase', ''], sep=';', skiprows=[0])
data['time'] = data['time'].apply(pd.to_datetime)

data_folder = Path(os.path.normpath(DATA_PATH + '/data/Measurements/Measurement3'))
dataframe_middle = pd.DataFrame()
for f in data_folder.iterdir():
	file_name = f.stem
	data_name = file_name[ file_name.find('m_s^2_') + 6 :]
	
	df = pd.read_csv(f, names=['time', 'amplitude', 'phase', ''], sep=';', skiprows=[0])
	
	df = df[1:]
	dataframe_middle[data_name] = df['amplitude']
  
""" Creating mean and median """
dataframe_temp = dataframe_middle.copy()
dataframe_clean = pd.DataFrame()
dataframe_clean['median'] = dataframe_temp.median(axis=1)
dataframe_clean['mean'] = dataframe_temp.mean(axis=1)
#dataframe_middle['median'] = dataframe_temp.median(axis=1)
#dataframe_middle['mean'] = dataframe_temp.mean(axis=1)

dataframe_middle = dataframe_middle.apply(pd.to_numeric)

dataframe_middle['timestamp'] = data['time']

""" Create a test dataframe with only one data column """

df_test = pd.DataFrame()
df_test['timestamp'] = dataframe_middle['timestamp']
df_test['value'] = dataframe_middle['2021_11_17_13_36_25_420']
#df_test['timestamp'] = df_test['timestamp'].apply(mpl_dates.date2num)
df_test['timestamp'] = (df_test.timestamp.view('int64') // 10**9 ) + datetime.now().timestamp()
#print(dataframe_middle.head())
#exit()
""" Preprocessing for training """
training_mean = df_test.mean()
training_std = df_test.std()
df_training_value = (df_test - training_mean) / training_std

""" Creating sequences """
TIME_STEPS = 288

# Generated training sequences for use in the model.
def create_sequences(values, time_steps=TIME_STEPS):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)


x_train = create_sequences(df_training_value.values)
#x_train = np.asarray(x_train).astype(np.float32)
print("Training input shape: ", x_train.shape)

""" Create Model """

from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential(
    [
        layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
        layers.Conv1D(
            filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Dropout(rate=0.2),
        layers.Conv1D(
            filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Conv1DTranspose(
            filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Dropout(rate=0.2),
        layers.Conv1DTranspose(
            filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
    ]
)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
print(model.summary())

""" Create Callbacks for Model """
model_name = 'Autoencoder' + datetime.now().strftime("%m%d%Y%H:%M")
checkpointer = ModelCheckpoint(os.path.join(DATA_PATH, "tensor_results", model_name + ".h5"), save_weights_only=True, save_best_only=True, verbose=1)
tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))

""" Training Model """

history = model.fit(
    x_train,
    x_train,
    epochs=50,
    batch_size=128,
    validation_split=0.1,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min"),
		checkpointer
    ],
)