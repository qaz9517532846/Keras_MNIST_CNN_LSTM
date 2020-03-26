import keras
import numpy as np
from keras.layers import LSTM, Conv1D, MaxPooling1D
from keras.layers import Dense, Activation
from keras.datasets import mnist
from keras.models import Sequential
from keras.optimizers import Adam

learning_rate = 0.001
epochs = 5
batch_size = 128
display_step = 10

img_rows = 28
img_cols = 28
n_hidden = 128
n_classes = 10

# the data, split between train and test sets
train_data = np.load('train_data.npy') 
test_data = np.load('test_data.npy') 

train_x = [i[0] for i in train_data]
print(np.shape(train_x))

train_y = [i[1] for i in train_data]
print(np.shape(train_y))

test_x = [i[0] for i in test_data]
print(np.shape(test_x))

test_y = [i[1] for i in test_data]
print(np.shape(test_y))

train_x = np.array(train_x, dtype = np.float32)
train_y = np.array(train_y, dtype = np.float32)
test_x = np.array(test_x, dtype = np.float32)
test_y = np.array(test_y, dtype = np.float32)

model = Sequential()
model.add(Conv1D(32, kernel_size = 3, activation = 'relu', input_shape = (img_rows, img_cols)))
model.add(Conv1D(64, 3, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(MaxPooling1D(3))
model.add(LSTM(n_hidden))
model.add(Dense(n_classes))
model.add(Activation('softmax'))

adam = Adam(lr=learning_rate)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer = adam,
              metrics=['accuracy'])

result = model.fit(train_x, train_y, batch_size = batch_size, epochs = epochs, verbose = 1)

score = model.evaluate(test_x, test_y, verbose = 0)

print('CNN & LSTM test score:', score[0])
print('CNN & LSTM test accuracy:', score[1])
