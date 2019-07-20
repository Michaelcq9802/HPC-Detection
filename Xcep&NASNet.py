#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Imports
import numpy as np 
import pandas as pd 
from glob import glob 
from skimage.io import imread 
import os
import shutil
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.nasnet import NASNetMobile
from keras.applications.xception import Xception
from keras.utils.vis_utils import plot_model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Average, Input, Concatenate, GlobalMaxPooling2D
from keras.models import Model
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.optimizers import Adam
get_ipython().system('pip install livelossplot')
from livelossplot import PlotLossesKeras


# In[10]:


TRAINING_LOGS_FILE = "training_logs.csv"
MODEL_SUMMARY_FILE = "model_summary.txt"
MODEL_FILE = "model.h5"
KAGGLE_SUBMISSION_FILE = "kaggle_submission.csv"


# In[11]:


input_dir = '/users/michael.qi_chen/downloads/'
training_dir = input_dir + 'train/'
data_frame = pd.DataFrame({'path': glob('/users/michael.qi_chen/downloads/train/*.tif')})
data_frame['id'] = data_frame.path.map(lambda x: x.split('/')[5].split('.')[0])
labels = pd.read_csv(input_dir + 'train_labels.csv')
data_frame = data_frame.merge(labels, on = 'id')
negatives = data_frame[data_frame.label == 0].sample(85000)
positives = data_frame[data_frame.label == 1].sample(85000)
data_frame = pd.concat([negatives, positives]).reset_index()
data_frame = data_frame[['path', 'id', 'label']]
data_frame['image'] = data_frame['path'].map(imread)

training_path = '../training'
validation_path = '../validation'

for folder in [training_path, validation_path]:
    for subfolder in ['0', '1']:
        path = os.path.join(folder, subfolder)
        os.makedirs(path, exist_ok=True)

training, validation = train_test_split(data_frame, train_size=0.9, stratify=data_frame['label'])

data_frame.set_index('id', inplace=True)

for images_and_path in [(training, training_path), (validation, validation_path)]:
    images = images_and_path[0]
    path = images_and_path[1]
    for image in images['id'].values:
        file_name = image + '.tif'
        label = str(data_frame.loc[image,'label'])
        destination = os.path.join(path, label, file_name)
        if not os.path.exists(destination):
            source = os.path.join(input_dir + 'train', file_name)
            shutil.copyfile(source, destination)


# In[12]:


# Data augmentation
training_data_generator = ImageDataGenerator(rescale=1./255,
                                             horizontal_flip=True,
                                             vertical_flip=True,
                                             rotation_range=90,
                                             zoom_range=0.2, 
                                             width_shift_range=0.1,
                                             height_shift_range=0.1,
                                             shear_range=0.05,
                                             channel_shift_range=0.1)


# In[13]:


# Data generation
training_generator = training_data_generator.flow_from_directory(training_path,
                                                                 target_size=(96,96),
                                                                 batch_size=216,
                                                                 class_mode='binary')
validation_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(validation_path,
                                                                              target_size=(96,96),
                                                                              batch_size=216,
                                                                              class_mode='binary')
testing_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(validation_path,
                                                                           target_size=(96,96),
                                                                           batch_size=216,
                                                                           class_mode='binary',
                                                                           shuffle=False)


# In[14]:


# Model
in_shape = (96, 96, 3)
inputs = Input(in_shape)
xception = Xception(include_top = False, weights = None, input_shape = in_shape)  
nas_net = NASNetMobile(include_top = False, weights = None, input_shape = in_shape)

outputs = Concatenate(axis=-1)([GlobalAveragePooling2D()(xception(inputs)),
                                GlobalAveragePooling2D()(nas_net(inputs))])
outputs = Dropout(0.5)(outputs)
outputs = Dense(1, activation='sigmoid')(outputs)
model = Model(inputs, outputs)
model.compile(optimizer=Adam(lr=0.0001, decay=0.00001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()


# In[ ]:


#  Training
history = model.fit_generator(training_generator,
                              steps_per_epoch=len(training_generator), 
                              validation_data=validation_generator,
                              validation_steps=len(validation_generator),
                              epochs=10,
                              verbose=1,
                              callbacks=[PlotLossesKeras(),ModelCheckpoint(MODEL_FILE,
                                                                             monitor='val_acc',
                                                                             verbose=1,
                                                                             save_best_only=True,
                                                            mode='max'),CSVLogger(TRAINING_LOGS_FILE,
                                                                               append=False,
                                                                               separator=';')])


# In[20]:


# Kaggle
testing_files = glob('/users/michael.qi_chen/downloads/test/*.tif')
submission = pd.DataFrame()
for index in range(0, len(testing_files), 5000):
    data_frame = pd.DataFrame({'path': testing_files[index:index + 5000]})
    data_frame['id'] = data_frame.path.map(lambda x: x.split('/')[5].split(".")[0])
    data_frame['image'] = data_frame['path'].map(imread)
    images = np.stack(data_frame.image, axis=0)
    predicted_labels = [model.predict(np.expand_dims(image/255.0, axis=0))[0][0] for image in images]
    predictions = np.array(predicted_labels)
    data_frame['label'] = predictions
    submission = pd.concat([submission, data_frame[["id", "label"]]])
submission.to_csv(KAGGLE_SUBMISSION_FILE, index=False, header=True)


# In[ ]:




