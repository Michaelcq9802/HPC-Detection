import numpy as np
import pandas as pd
import matplotlib
import os
import matplotlib.pyplot as plt
from matplotlib import patches
import cv2
from keras.callbacks import Callback
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import roc_curve,auc, confusion_matrix


print ("Number of train files:",len(os.listdir("F:\\桌面\\project\\input\\train\\")))
print ("Number of test files:",len(os.listdir("F:\\桌面\\project\\input\\test\\")))

dftrain=pd.read_csv("train_labels.csv",dtype=str)
dftrain.head()


print("Counts of negative and postive labels in training data:")
dftrain.groupby(['label']).count()


def add_ext(id):
    return id+".tif"

dftrain["id"]=dftrain["id"].apply(add_ext)

def addpath(col):
    return 'F:\\桌面\\project\\input\\train\\/' + col 

dftrain['Path']=dftrain['id'].apply(addpath)

dftrain.head()


datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.2)


batch_size = 20
image_size = (96,96)

train_generator=datagen.flow_from_dataframe(
dataframe=dftrain,
directory="F:\\桌面\\project\\input\\train\\",
x_col="id",
y_col="label",
subset="training",
batch_size=batch_size,
seed=42,
shuffle=True,
class_mode="binary",
target_size=image_size)

validation_generator=datagen.flow_from_dataframe(
dataframe=dftrain,
directory="F:\\桌面\\project\\input\\train\\",
x_col="id",
y_col="label",
subset="validation",
batch_size=batch_size,
seed=42,
shuffle=True,
class_mode="binary",
target_size=image_size)


kernel_size=(3,3)
pool_size=(2,2)
first_filter=32
second_filter=64
third_filter=128

dropout_conv=0.3
dropout_dense=0.3

model = Sequential()
model.add(Conv2D(first_filter, kernel_size, activation='relu', input_shape= (96,96,3)))
model.add(Conv2D(first_filter, kernel_size, use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(dropout_conv))

model.add(Conv2D(second_filter, kernel_size, use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(second_filter, kernel_size, use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))

model.add(Conv2D(third_filter, kernel_size, use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Conv2D(third_filter, kernel_size, use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))

model.add(Flatten())
model.add(Dense(256, use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(dropout_dense))
model.add(Dense(1, activation = "sigmoid"))

model.compile(Adam(0.01), loss = "binary_crossentropy", metrics=["accuracy"])
model.summary()


trainstep=train_generator.n//train_generator.batch_size
valstep=validation_generator.n//validation_generator.batch_size

filepath="weights-best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

history=model.fit_generator(generator=train_generator,
                    steps_per_epoch=trainstep,
                    validation_data=validation_generator,
                    validation_steps=valstep,
                    epochs=10,
                    callbacks=[checkpoint]
)


model.load_weights(filepath) #load saved weights
test_datagen=ImageDataGenerator(rescale=1./255)


test_generator=datagen.flow_from_dataframe(
dataframe=dftrain,
directory="F:\\桌面\\project\\input\\train\\",
x_col="id",
y_col="label",
subset="validation",
batch_size=5,   
seed=42,
shuffle=False,  
class_mode="binary",
target_size=image_size)


test_labels = test_generator.classes
y_preds = model.predict_generator(test_generator,verbose=1,steps=test_generator.n/5)
y_pred_keras=y_preds.round()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_labels, y_pred_keras)
auc_keras = auc(fpr_keras, tpr_keras)

print('AUC score :', + auc_keras)


test_results=pd.DataFrame({'id':os.listdir("F:\\桌面\\project\\input\\test\\")})
test_datagen=ImageDataGenerator(rescale=1./255)

submit_generator=datagen.flow_from_dataframe(
dataframe=test_results,
directory="F:\\桌面\\project\\input\\test\\",
x_col="id",
batch_size=2,   
shuffle=False,  
class_mode=None,
target_size=image_size)


y_test_prob=model.predict_generator(submit_generator,verbose=1,steps=submit_generator.n/2)
y_test_pred=y_test_prob.round()
def remove_ext(id):
    return (id.split('.'))[0]
test_results['id']=test_results['id'].apply(remove_ext)


test_results['label'] = y_test_pred
test_results.to_csv("sequential_submission.csv",index=False)
test_results.head()