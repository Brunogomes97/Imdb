import numpy as np
import functions as ft
import pandas as pd
from keras.optimizers import SGD #,Adam 
from keras.utils import np_utils
from wide_resnet import WideResNet
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from mixup_generator import MixupGenerator
from random_eraser import get_random_eraser
import time


aug = True
namecall =  "imdbWRN_64-{}".format(int(time.time()))
output_path = 'checkpoints/imdbase/'
batch_size = 32
nb_epochs = 30

class Schedule:
    def __init__(self, nb_epochs, initial_lr):
        self.epochs = nb_epochs
        self.initial_lr = initial_lr

    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.25:
            return self.initial_lr
        elif epoch_idx < self.epochs * 0.50:
            return self.initial_lr * 0.2
        elif epoch_idx < self.epochs * 0.75:
            return self.initial_lr * 0.04
        return self.initial_lr * 0.008




# Load Base
image, gender, age  = ft.load_data("serializer/MATimdb64.mat")

x_data = image
y_data_g = np_utils.to_categorical(gender,2)
y_data_a = np_utils.to_categorical(age,101)

# Fatores importantes: 
# optimizers, largura da rede, profundidade da rede, losses, val_split,

# model WidesResNet (WRN) dupla saida(age,gender)
model = WideResNet(64, depth=16, k=8)()
#model.summary()



#optimizers
model.compile(SGD(lr=0.1, momentum=0.9, nesterov=True), loss = ['categorical_crossentropy','categorical_crossentropy'],
              metrics = ['accuracy'])


#Pre-process Base
validation_split = 0.1


data_num = len(x_data)
#Randomizar base
indexes = np.arange(data_num)
np.random.shuffle(indexes)
x_data = x_data[indexes]

y_data_g = y_data_g[indexes] #Gender
y_data_a = y_data_a[indexes] #Age
train_num = int(data_num * (1 - validation_split))

x_train = x_data[:train_num] #Treino Img
x_test = x_data[train_num:] #Teste Img
y_train_g = y_data_g[:train_num] #Treino gender
y_test_g = y_data_g[train_num:] #Teste gender
y_train_a = y_data_a[:train_num] #Treino age
y_test_a = y_data_a[train_num:] #treino age



callbacks = [LearningRateScheduler(schedule=Schedule(nb_epochs, 0.1)), #Callbacks
                 ModelCheckpoint(str(output_path) + "/weights.{epoch:02d}-{val_loss:.2f}.hdf5",
                                 monitor="val_loss",
                                 verbose=1,
                                 save_best_only=True,
                                 mode="auto"),TensorBoard(log_dir='TensorBoard/imdb-64'.format(namecall))
                 ]


if aug:
    datagen = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            preprocessing_function=get_random_eraser(v_l=0, v_h=255))
    
    training_generator = MixupGenerator(x_train, [y_train_g, y_train_a], batch_size=32, alpha=0.2,
                                         datagen=datagen)()
    
    hist = model.fit_generator(generator=training_generator,
                                   steps_per_epoch=train_num // batch_size,
                                   validation_data=(x_test, [y_test_g, y_test_a]),
                                   epochs=nb_epochs, verbose=1,
                                   callbacks=callbacks)
else:
    hist = model.fit(x_train, [y_train_g, y_train_a], batch_size=32, epochs=30,
                         validation_data=(x_test, [y_test_g, y_test_a]))
    

pd.DataFrame(hist.history).to_hdf(output_path.joinpath("history-imdb64.h5", "history"))
