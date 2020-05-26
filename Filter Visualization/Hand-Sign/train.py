import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, Flatten, Dropout,Conv2D, MaxPooling2D, BatchNormalization
from keras.utils.np_utils import to_categorical
path = 'data'
myImg = os.listdir(path)
images = []
images_label = []

for x in range(len(myImg)):
    cur_dir = os.listdir(path+"/"+str(x))
    for y in cur_dir:
        cur_img = cv2.imread(path+"/"+str(x)+"/"+y)
        images.append(cur_img)
        images_label.append(x)
    print(x, end=" ")
print(" ")

images = np.array(images)
images_label = np.array(images_label)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train, y_test = train_test_split(images,images_label,test_size=0.2,
                                                 random_state=0)
distribution_train=[]
distribution_test=[]
import matplotlib as mpl
mpl.use('Qt5Agg')
for x in range(len(myImg)):
    distribution_train.append(len(np.where(y_train==x)[0]))
    distribution_test.append(len(np.where(y_test == x)[0]))

plt.figure(figsize=(10,10))
plt.bar(range(0,len(myImg)), distribution_train)
plt.xticks(np.arange(0, len(myImg), step=1))
plt.show()

plt.figure(figsize=(10,10))
plt.bar(range(0,len(myImg)), distribution_test)
plt.xticks(np.arange(0, len(myImg), step=1))
plt.show()

def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _,img = cv2.threshold(img, 130,255, cv2.THRESH_BINARY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

#dummy = preProcessing(x_train[15])
#cv2.imshow("example", dummy)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

x_train = np.array(list(map(preProcessing, x_train)))
x_test = np.array(list(map(preProcessing, x_test)))

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)

from keras.preprocessing.image import ImageDataGenerator
batch_size=32
data_generator = ImageDataGenerator(width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    shear_range=0.1,
                                    rotation_range=10)
data_generator.fit(x_train)
def myModel():
    i = Input(shape=(x_train[0].shape))
    x = Conv2D(32,(3,3),padding='same',activation='relu')(i)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)

    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    #x = BatchNormalization()(x)
    x = MaxPooling2D()(x)

    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    #x = BatchNormalization()(x)
    x = MaxPooling2D()(x)

    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    #x = BatchNormalization()(x)
    x = MaxPooling2D()(x)

    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)

    x = Flatten()(x)
    x = Dropout(0.2)(x)
    x= Dense(512,activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(len(myImg), activation='softmax')(x)

    model = Model(i,x)
    model.compile(Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = myModel()
print(model.summary())
train_generator = data_generator.flow(x_train, y_train, batch_size)
steps_per_epoch = x_train.shape[0]//batch_size
r = model.fit(train_generator, validation_data=(x_test, y_test),steps_per_epoch=steps_per_epoch, epochs=50)

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(r.history['accuracy'], label='accuracy')
plt.plot(r.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()
score = model.evaluate(x_test, y_test)
print(score[0])
print(score[1])

import pickle
pickle_out = open("all_files/model_trained.p","wb")
pickle.dump(model,pickle_out)
pickle_out.close()
