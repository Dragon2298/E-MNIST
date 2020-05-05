#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import idx2numpy


# In[4]:


train_images = open('data/emnist-byclass-train-images-idx3-ubyte','rb')
train_labels = open('data/emnist-byclass-train-labels-idx1-ubyte','rb')


# In[5]:


test_images = open('data/emnist-byclass-test-images-idx3-ubyte','rb')
test_labels = open('data/emnist-byclass-test-labels-idx1-ubyte','rb')


# In[6]:


train_images = idx2numpy.convert_from_file(train_images)
test_images = idx2numpy.convert_from_file(test_images)
train_labels = idx2numpy.convert_from_file(train_labels)
test_labels = idx2numpy.convert_from_file(test_labels)


# In[7]:


train_images = train_images/255
test_images = test_images/255


# In[9]:


train_images = train_images.reshape(697932,28,28,1)


# In[11]:


test_images = test_images.reshape(116323,28,28,1)


# In[13]:


train_labels = train_labels.reshape(697932,1)
test_labels = test_labels.reshape(116323,1)


# In[18]:


from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss',patience=2)


# In[14]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten


# In[15]:


model = Sequential()

model.add(Conv2D(64,(3,3),activation='relu',input_shape = (28,28,1)))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dense(62,activation = 'softmax'))


# In[16]:


model.compile(loss= 'sparse_categorical_crossentropy',
             optimizer = 'adam',
             metrics=['accuracy'])


# In[19]:


model.fit(train_images,train_labels,epochs = 7,validation_data=(test_images,test_labels),
         callbacks =[early_stop])


# In[20]:


losses = pd.DataFrame(model.history.history)



# In[29]:


pred = model.predict_classes(test_images)


# In[30]:


from sklearn.metrics import classification_report,confusion_matrix


# In[31]:

print(model.evaluate(test_images,test_labels))
print(classification_report(test_labels,pred),'\n\n',confusion_matrix(test_labels,pred))
losses[['loss','val_loss']].plot()

from tensorflow.keras.models import load_model

model.save('emnist.h5')



