import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import sleep
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
import os
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.layers import Dropout
from keras.layers import MaxPooling2D, Conv2D , Flatten, Dropout
from keras.layers.normalization import BatchNormalization

train_data = pd.read_csv('D:/Prabhkirat/Python/Facial/training.csv')
test_data = pd.read_csv('D:/Prabhkirat/Python/Facial/test.csv')
lookid_data = pd.read_csv('D:/Prabhkirat/Python/Facial/IdLookupTable.csv')
#print(train_data.columns)
print(train_data.isnull().any().value_counts())
train_data.fillna(method = 'ffill',inplace = True)
print(train_data.isnull().any().value_counts())


def format_img(x):
    return np.asarray([int(e) for e in x.split(' ')], dtype=np.uint8).reshape(96,96)

with Parallel(n_jobs=10, verbose=1, prefer='threads') as ex:
    x = ex(delayed(format_img)(e) for e in train_data.Image)
    
X_train = np.stack(x)[..., None]
print(X_train.shape)
y = train_data.iloc[:, :-1].values
print(train_data.iloc[:, :-1].values)

def show(x, y=None):
    plt.imshow(x[..., 0], 'gray')
    if y is not None:
        points = np.vstack(np.split(y, 15)).T
        plt.plot(points[0], points[1], 'o', color='red')
        
    plt.axis('off')

sample_idx = np.random.choice(len(X_train))    
show(X_train[sample_idx], y[sample_idx])
plt.show()

x_train, x_test, y_train, y_test = train_test_split(X_train, y, test_size=0.2, random_state=42)
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

model = Sequential()
model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(96,96,1)))
model.add(Dropout(0.1))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode="valid"))
model.add(BatchNormalization())

model.add(Conv2D(32, 5, 5,activation="relu"))
    # model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode="valid"))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(64, 5, 5,activation="relu"))
    # model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode="valid"))
model.add(BatchNormalization())

model.add(Conv2D(128, 3, 3,activation="relu"))
    # model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode="valid"))
model.add(Dropout(0.4))
model.add(BatchNormalization())

model.add(Flatten())

model.add(Dense(500, activation="relu"))
model.add(Dropout(0.1))

model.add(Dense(128, activation="relu"))
model.add(Dropout(0.1))

model.add(Dense(30))


model.summary()
model.compile(optimizer='adam', loss='mse', metrics=['mae','accuracy'])

hist2 = model.fit(x_train, y_train, nb_epoch=150,batch_size=256, validation_split=0.2, verbose = 2)

def format_img(x):
    return np.asarray([int(e) for e in x.split(' ')], dtype=np.uint8).reshape(96,96)

with Parallel(n_jobs=10, verbose=1, prefer='threads') as ex:
    y = ex(delayed(format_img)(e) for e in test_data.Image)
	
pred_y = np.stack(y)[..., None]
print(pred_y.shape)

pred = model.predict(pred_y)
y_pred = pd.DataFrame(pred)
y_pred.to_csv('d:/Prabhkirat/Python/facial/facial_pred.csv', index=False)

