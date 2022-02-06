import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


#rint(arr.shape)

# test = arr[1:-2,:]
# print(test)

#each array consists of a timestamp, the l_shoulderAngles(XY,YZ,XZ) and the r_shoulderAngles
arms_front_array = np.empty((0, 3))
arms_horizontal_array = np.empty((0, 3))
arms_side_array = np.empty((0, 3))
arms_sky_array = np.empty((0, 3))


#rearange arrays since it's strongly nested
def rearange(_listArray):
    _new_array = np.empty((len(_listArray), 7))
    for r in range(len(_new_array)):

        _new_array[r, 0] = _listArray[r][0]

        for x in range(3):
            _new_array[r, x + 1] = _listArray[r][1][0][x]

        for x in range(3):
            _new_array[r, x + 4] = _listArray[r][2][0][x]

    return _new_array


# region load arrays

#allowing pickle is false by default and prevents loading the numpy array
#first two seconds (30*2) are cut away as well as the last second to not include false positions


arms_side_01 = np.load('arms_side_standard_01_output.npy', allow_pickle=True)
arms_side_02 = np.load('arms_side_standard_02_output.npy', allow_pickle=True)
#arms_side_03 = np.load('03_arms_side_slow_output.npy', allow_pickle=True)


arms_side_array = np.append(arms_front_array, arms_side_01, axis=0)
arms_side_array = np.append(arms_front_array, arms_side_02, axis=0)
#arms_side_array = np.append(arms_front_array, arms_side_03[60:-30,:], axis=0)

_arms_side_array = rearange(arms_side_array)

arms_sky_01 = np.load('01_arms_sky_frontView_output.npy', allow_pickle=True)
arms_sky_02 = np.load('02_arms_sky_speedWork_output.npy', allow_pickle=True)
arms_sky_03 = np.load('03_arms_sky_slow_output.npy', allow_pickle=True)
arms_sky_04 = np.load('04_arms_sky_highAngle_output.npy', allow_pickle=True)

arms_sky_array = np.append(arms_front_array, arms_sky_01[60:-30,:], axis=0)
arms_sky_array = np.append(arms_front_array, arms_sky_02[60:-30,:], axis=0)
arms_sky_array = np.append(arms_front_array, arms_sky_03[60:-30,:], axis=0)
arms_sky_array = np.append(arms_front_array, arms_sky_04[60:-30,:], axis=0)

_arms_sky_array = rearange(arms_sky_array)

# endregion -

# print(arms_horizontal_array[0:5])
# print(_arms_horizontal_array[0:5])
# print(_arms_horizontal_array.shape)
# print(_arms_side_array.shape)
# print(_arms_sky_array.shape)


#Create dataframe


# data = pd.DataFrame(data = processedList, columns = columns)
# data.head()

#creating individual dataframes per exercise
columns = ['frame', 'l_XY', 'l_YZ', 'l_XZ', 'r_XY', 'r_YZ', 'r_XZ']

arms_front_df = pd.DataFrame(data = _arms_front_array, columns = columns)
arms_front_df.insert(0,'exercise','arms_front')

arms_side_df = pd.DataFrame(data = _arms_side_array, columns = columns)
arms_side_df.insert(0,'exercise','arms_side')

arms_horizontal_df = pd.DataFrame(data = _arms_horizontal_array, columns = columns)
arms_horizontal_df.insert(0,'exercise','arms_horizontal')

arms_sky_df = pd.DataFrame(data = _arms_sky_array, columns = columns)
arms_sky_df.insert(0,'exercise','arms_sky')

#merge table to one
frames = [arms_front_df, arms_side_df, arms_horizontal_df,arms_sky_df ]
exerciseTable = pd.concat(frames)


#drop the timestamp
df = exerciseTable.drop(['frame'], axis = 1).copy()

#investigate exercise distribution
print(df['exercise'].value_counts())

#Balance data - all max 4607
arms_front = df[df['exercise']=='arms_front'].head(4607).copy()
arms_horizontal = df[df['exercise']=='arms_horizontal'].head(4607).copy()
arms_side = df[df['exercise']=='arms_side'].head(4607).copy()
arms_sky = df[df['exercise']=='arms_sky'].head(4607).copy()

balanced_data = pd.DataFrame()
balanced_data = balanced_data.append([arms_front, arms_horizontal, arms_side, arms_sky])

#data is now balanced
print(balanced_data['exercise'].value_counts())
print(balanced_data.shape)

#encode labels - use label.class_ to recover the mapped values to the label numbers
label = LabelEncoder()
balanced_data['label'] = label.fit_transform(balanced_data['exercise'])
print(balanced_data.head())

#Standardization of data - don't know whether I need that

X = balanced_data[['l_XY', 'l_YZ', 'l_XZ', 'r_XY', 'r_YZ', 'r_XZ']]
y = balanced_data['label']

scaler = StandardScaler()
X = scaler.fit_transform(X)

scaled_X = pd.DataFrame(data = X, columns = ['l_XY', 'l_YZ', 'l_XZ', 'r_XY', 'r_YZ', 'r_XZ'])
scaled_X['label'] = y.values

print(scaled_X.head())

#Frame preparation - we take two seconds of data and then move one second -> overlapping

import scipy.stats as stats
Fs = 30
frame_size = Fs*2 # 60
hop_size = Fs*1 # 30


def get_frames(df, frame_size, hop_size):
    N_FEATURES = 6

    frames = []
    labels = []
    for i in range(0, len(df) - frame_size, hop_size):
        l_XY = df['l_XY'].values[i: i + frame_size]
        l_YZ = df['l_YZ'].values[i: i + frame_size]
        l_XZ = df['l_XZ'].values[i: i + frame_size]
        r_XY = df['r_XY'].values[i: i + frame_size]
        r_YZ = df['r_YZ'].values[i: i + frame_size]
        r_XZ = df['r_XZ'].values[i: i + frame_size]

        # Retrieve the most often used label in this segment
        label = stats.mode(df['label'][i: i + frame_size])[0][0]
        frames.append([l_XY, l_YZ, l_XZ, r_XY, r_YZ, r_XZ])
        labels.append(label)

    # Bring the segments into a better shape
    frames = np.asarray(frames).reshape(-1, frame_size, N_FEATURES)
    labels = np.asarray(labels)

    return frames, labels


X, y = get_frames(scaled_X, frame_size, hop_size)

print(X.shape, y.shape)


#Dividing into trainining & testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)
print(X_train.shape, X_test.shape)
    #reshape to make it acceptable for the CNN
X_train = X_train.reshape(490, 60, 6, 1)
X_test = X_test.reshape(123, 60, 6, 1)
print(X_train[0].shape, X_test[0].shape)


#CNN

model = Sequential()
model.add(Conv2D(16, (2, 2), activation = 'relu', input_shape = X_train[0].shape))
model.add(Dropout(0.1))

model.add(Conv2D(32, (2, 2), activation='relu'))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(4, activation='softmax'))

#compiling model
model.compile(optimizer=Adam(learning_rate = 0.001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(X_train, y_train, epochs = 15, validation_data= (X_test, y_test), verbose=1)


#Plot the results
def plot_learningCurve(history, epochs):
  # Plot training & validation accuracy values
  epoch_range = range(1, epochs+1)
  plt.plot(epoch_range, history.history['accuracy'])
  plt.plot(epoch_range, history.history['val_accuracy'])
  plt.title('Model accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

  # Plot training & validation loss values
  plt.plot(epoch_range, history.history['loss'])
  plt.plot(epoch_range, history.history['val_loss'])
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

plot_learningCurve(history, 15)


#confusion matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix


y_pred = model.predict_classes(X_test)
mat = confusion_matrix(y_test, y_pred)
fig, ax = plot_confusion_matrix(conf_mat=mat, class_names=label.classes_, show_normed=False, figsize=(4,4))
plt.show()


#arms_horizontal_array = np.append(arms_front_array, arms_front_01, axis=0)

