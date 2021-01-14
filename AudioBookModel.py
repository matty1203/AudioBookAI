# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 23:40:43 2021

@author: Mathews
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
import tensorflow as tf

raw_data=np.loadtxt('Audiobooks_data.csv',delimiter=',')

unscaled_ip=raw_data[:,1:-1]
targets_all=raw_data[:,-1]

###Data Balancing
true_targets_cnt=np.sum(targets_all==1)
false_targets_cnt=0
ind_to_rm=[]

for i in range(targets_all.shape[0]):
    if targets_all[i]==0:
        false_targets_cnt+=1
        if false_targets_cnt>true_targets_cnt:
            ind_to_rm.append(i)

balanced_unscaled_inputs = np.delete(unscaled_ip,ind_to_rm,axis=0)
balanced_targets=np.delete(targets_all,ind_to_rm,axis=0)


####Standardising Data

balanced_scaled_ip=preprocessing.StandardScaler().fit_transform(balanced_unscaled_inputs)

####Shuffle Data

#shuffle
shuffled_ind= np.arange(balanced_scaled_ip.shape[0])
np.random.shuffle(shuffled_ind)
shuffled_ip=balanced_scaled_ip[shuffled_ind]
shuffled_targets=balanced_targets[shuffled_ind]

###Train Test Split

samples_count=shuffled_ip.shape[0]
train_sample_size=int(0.8*samples_count)
validation_sample_size=int(0.1*samples_count)
test_sample_size=samples_count-train_sample_size-validation_sample_size

train_ip=shuffled_ip[:train_sample_size]
train_targets=shuffled_targets[:train_sample_size]


val_ip=shuffled_ip[train_sample_size:(validation_sample_size+train_sample_size)]
val_targets=shuffled_targets[train_sample_size:(validation_sample_size+train_sample_size)]

test_ip=shuffled_ip[(validation_sample_size+train_sample_size):]
test_targets=shuffled_targets[(validation_sample_size+train_sample_size):]

##saving as .npz file

np.savez('Audiobooks_data_train',inputs=train_ip,targets=train_targets)
np.savez('Audiobooks_data_val',inputs=val_ip,targets=val_targets)
np.savez('Audiobooks_data_test',inputs=test_ip,targets=test_targets)


########Neural Network Design

##Train Data
data_train=np.load('Audiobooks_data_train.npz')
data_train_ip=data_train['inputs'].astype(np.float)
data_train_targets=data_train['targets'].astype(np.int)

##Validation Data
data_val=np.load('Audiobooks_data_val.npz')
data_val_ip=data_val['inputs'].astype(np.float)
data_val_targets=data_val['targets'].astype(np.int)


###Test Data
data_test=np.load('Audiobooks_data_test.npz')
data_test_ip=data_test['inputs'].astype(np.float)
data_test_targets=data_test['targets'].astype(np.int)

##Neural Network Creation
IP_SIZE=10
HIDDEN_LAYER_SIZE=50
OP_SIZE=2

model=tf.keras.Sequential([
                           tf.keras.layers.Dense(HIDDEN_LAYER_SIZE,activation='relu'),
                           tf.keras.layers.Dense(HIDDEN_LAYER_SIZE,activation='relu'),
                           tf.keras.layers.Dense(OP_SIZE,activation='softmax'),
    ])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

BATCH_SIZE=100
EPOCHS=100
early_stopping=tf.keras.callbacks.EarlyStopping(patience=2)
model.fit(data_train_ip,
          data_train_targets,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=2,
          callbacks=[early_stopping],
          validation_data=(data_val_ip,data_val_targets)
          )

##Test Model

test_loss,test_acc=model.evaluate(data_test_ip,data_test_targets)

print("Loss:",test_loss)
print("Acc:",test_acc)
