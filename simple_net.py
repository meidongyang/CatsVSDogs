#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-3-07 17:35:39
# @Author  : Mei
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, Dropout, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model

ROOT_DIR = os.getcwd()
BATCH_SIZE = 16

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation('tanh'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# 数据预处理：数据增强
train_data_gen = ImageDataGenerator(rotation_range=40,  # 图片旋转角度
                                    width_shift_range=0.2,  # 图片水平（宽度方向）偏移幅度
                                    height_shift_range=0.2,  # 图片竖直（高度方向）偏移幅度
                                    rescale=1./255,  # 重缩放因子，将该数值乘到数据上
                                    shear_range=0.2,  # 剪切强度。逆时针方向剪切变换角度
                                    zoom_range=0.2,  # 随机缩放幅度。[lower,upper] = [1 - zoom_range, 1+zoom_range]
                                    horizontal_flip=True,  # 随机水平翻转
                                    fill_mode='nearest')  # 处理变换时超出边界的点
test_data_gen = ImageDataGenerator(rescale=1./255)

# 创建数据生成器-训练集
train_data_path = os.path.join(ROOT_DIR, 'train_data')
trian_generator = train_data_gen.flow_from_directory(train_data_path,  # 训练集所在目录
                                                     target_size=(150, 150),  # 所有图片调整大小为150 * 150
                                                     batch_size=BATCH_SIZE,
                                                     class_mode='binary')  # 返回1D的二值标签
# 同样创建一个验证集的生成器
val_data_path = os.path.join(ROOT_DIR, 'validation_data')
val_generator = test_data_gen.flow_from_directory(val_data_path,
                                                  target_size=(150, 150),
                                                  batch_size=BATCH_SIZE,
                                                  class_mode='binary')

# 画出网络模型的结构图
plot_model(model, to_file='model.png')

# train model
model.fit_generator(trian_generator,
                    steps_per_epoch=100,
                    epochs=1,
                    validation_data=val_generator,
                    validation_steps=10)
model.save_weights('simple_net.h5')


