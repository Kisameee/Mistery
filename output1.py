import datetime

from keras import optimizers
from keras.callbacks import *
from keras.layers import Dense, Dropout
from keras.models import Sequential

from sequencer import *

# TODO: change name experiment and parameter every run !

ts = time.time()
date_time = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')

experiment_name = "model(Linear) epoch(1) dense(1020) lr(0.1) activation(relu + softmax)  " + date_time
checkpoint = ModelCheckpoint('/home/soat/Mystere/models_save/'f'{experiment_name}.h5',
                             monitor='val_binary_accuracy', verbose=1, save_best_only=True, mode='max')

print("Launching experiment : " + experiment_name)

# TODO: change the path !!!
inputData_path = '/home/soat/Mystere/data/2018_04_28_full_train-000000-input.npy'
outputData_path = '/home/soat/Mystere/data/2018_04_28_full_train-000000-output1.npy'
inputTest_path = '/home/soat/Mystere/data/2018_04_28_full_test-000000-input.npy'
outputTest_path = '/home/soat/Mystere/data/2018_04_28_full_test-000000-output1.npy'

logs_directory = "logs/"
models_directory = "models_save/"

input_size = 15444000
nb_classes = 1020
batch_size = 4096
input_shape = (1020,)
epochs = 1

# Loading the dataSet
# mmap_mode : {None, 'r+', 'r', 'w+', 'c'}, optional
inputData = np.load(inputData_path, mmap_mode='r+')
outputData = np.load(outputData_path, mmap_mode='r+')
inputTest = np.load(inputTest_path, mmap_mode='r+')
outputTest = np.load(outputTest_path, mmap_mode='r+')

# pre_process splitting train & val data
inputData_train, inputData_val = np.split(inputData, [int(.7 * len(inputData))])
outputData_train, outputData_val = np.split(outputData, [int(.7 * len(outputData))])

# take a look


print("inputData shape :" + str(inputData.shape) +
      " - outputData shape :" + str(outputData.shape))

print("*****************************************************")

print("inputData_train shape :" + str(inputData_train.shape) +
      " - inputData_val shape :" + str(inputData_val.shape))

print("outputData_train shape :" + str(outputData_train.shape) +
      " - outputData_val shape :" + str(outputData_val.shape))

# Generator
train_generator = Sequencer(inputData_train, outputData_train, nb_classes, batch_size, True)
val_generator = Sequencer(inputData_val, outputData_val, nb_classes, batch_size, True)
predict_generator = Sequencer(inputTest, outputTest, nb_classes, batch_size, True)

# Model
model = Sequential()
model.add(Dense(1020, activation='relu', input_shape=input_shape))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

Custom_SGD = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(Custom_SGD, loss='binary_crossentropy', metrics=['accuracy'])

# Callback
tb_callback = TensorBoard(logs_directory + experiment_name)
callback_list = [tb_callback, checkpoint]

# Fitting
model.fit_generator(
    train_generator,
    steps_per_epoch=input_size // batch_size,
    epochs=epochs,
    callbacks=callback_list,
    validation_data=val_generator)

trained_model = '/home/soat/Mystere/models_save/'f'{experiment_name}.h5'

zbeb = inputTest, outputTest
predictions = model.predict(zbeb[0], batch_size=batch_size)
good_result = 0
for p in range(len(predictions)):
    if np.argmax(predictions[p]) == np.argmax(zbeb[1][p]):
        good_result += 1
print('Precision :', str(good_result / len(zbeb[1])))
