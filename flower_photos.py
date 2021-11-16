#导入模块
import glob
import os
import cv2
import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


#设置初始化环境
path = './flower_photos/'
w = 100
h = 100
c = 3
my_epochs = 2


#构建读取图片数据集函数
def read_img(path):
    cate = [path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    imgs = []
    labels = []
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder+'/*.jpg'):
            img = cv2.imread(im)
            img = cv2.resize(img,(w,h))
            imgs.append(img)
            labels.append(idx)
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)


#读取图片数据集
data, label = read_img(path)
print("shape of data:", data.shape)
print("shape of label:", label.shape)


#划分训练集与测试集
seed = 785
np.random.seed(seed)
(x_train, x_val, y_train, y_val) = train_test_split(data, label, test_size=0.20, random_state=seed)
x_train = x_train / 255
x_val = x_val / 255
flower_dict = {0:'daisy', 1:'dandelion', 2:'roses', 3:'sunflowers', 4:'tulips'}


#构建CNN神经网络
model = Sequential([
    layers.Conv2D(32, kernel_size=[5, 5], padding= "same",activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
    layers.Dropout(0.25),
    
    layers.Conv2D(64, kernel_size=[3, 3], padding="same",activation=tf.nn.relu),
    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
    layers.Dropout(0.25),
    
    layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
    layers.Dropout(0.25),
    
    layers.Flatten(),
    layers.Dense(512, activation=tf.nn.relu),
    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(5, activation='softmax')
])


#构建adam优化器
opt = optimizers.Adam(lr=0.0001) # adam mse 交叉熵
model.compile(optimizer=opt,
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

#训练CNN图像识别模型
history = model.fit(x_train, y_train, epochs=my_epochs, validation_data=(x_val, y_val), batch_size=200, verbose=2)

model.summary()

def show_history(history,epochs):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(epochs)
    
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

show_history(history,my_epochs)

model.save('./Model/model.h5')

#验证集图片命名格式为test(i)
path_test = './testimages/'
imgs = []
for im in glob.glob(path_test+'/*.jpg'):
    print('reading the images:%s'%(im))
    img = cv2.imread(im)
    img = cv2.resize(img, (w, h))
    imgs.append(img)
imgs = np.asarray(imgs, np.float32)
print("shape of data:", imgs.shape)
prediction = model.predict_classes(imgs)
for i in range(np.size(prediction)):
    print("第",i+1,"朵花预测："+flower_dict[prediction[i]])
    img = plt.imread(path_test+"test"+str(i+1)+".jpg")
    plt.title('Maybe this is ' + flower_dict[prediction[i]] + " !")
    plt.imshow(img)
    plt.show()