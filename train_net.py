#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-3-07 17:30:55
# @Author  : Mei
from __future__ import print_function
import os
import numpy as np
import pandas as pd
from keras import callbacks
from keras.applications import DenseNet121, InceptionResNetV2, VGG19, ResNet50
from keras.models import Sequential, Model
from keras.utils import plot_model
from keras.optimizers import SGD
from keras.layers import Dense, Flatten, concatenate, merge
from keras.layers import Dropout, GlobalMaxPooling2D, Input
from keras.preprocessing.image import ImageDataGenerator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 数据集路径
ROOT_DIR = os.getcwd()  # 根目录路径
train_data_path = os.path.join(ROOT_DIR, 'train_data')
val_data_path = os.path.join(ROOT_DIR, 'validation_data')
test_data_path = os.path.join(ROOT_DIR, 'test1')

BATCH_SIZE = 32
EPOCHS = 5000
DATA_NUM = 22500
VAL_NUM = 2500
TEST_NUM = 12500

x_train_labels = []
x_val_labels = []

# 图片的宽、高
img_width, img_height = 150, 150

# 准备数据生成器的数据增强配置
train_data_gen = ImageDataGenerator(rotation_range=40,  # 图片旋转角度
                                    width_shift_range=0.2,  # 图片水平（宽度方向）偏移幅度
                                    height_shift_range=0.2,  # 图片竖直（高度方向）偏移幅度
                                    rescale=1./255,  # 重缩放因子，将该数值乘到数据上
                                    shear_range=0.2,  # 剪切强度。逆时针方向剪切变换角度
                                    zoom_range=0.2,  # 随机缩放幅度。[lower,upper] = [1 - zoom_range, 1+zoom_range]
                                    horizontal_flip=True,  # 随机水平翻转
                                    fill_mode='nearest')  # 处理变换时超出边界的点
test_data_gen = ImageDataGenerator(rescale=1./255)

train_generator = train_data_gen.flow_from_directory(train_data_path,
                                                     target_size=(img_height, img_width),
                                                     batch_size=BATCH_SIZE,
                                                     class_mode='binary')
val_generator = test_data_gen.flow_from_directory(val_data_path,
                                                  target_size=(img_height, img_width),
                                                  batch_size=BATCH_SIZE,
                                                  class_mode='binary')


main_input = Input((img_height, img_width, 3), dtype=np.float32)
# 构建不带分类器的预训练模型 - DenseNet121
dense_model = DenseNet121(input_tensor=main_input, weights='imagenet', include_top=False)
dense_x = GlobalMaxPooling2D()(dense_model.output)
print('DenseNet121 Model loaded.')

# 构建不带分类器的预训练模型 - InceptionResNetV2
# inception_res_model = InceptionResNetV2(input_tensor=main_input, weights='imagenet', include_top=False)
# inception_res_x = GlobalMaxPooling2D()(inception_res_model.output)
# print('InceptionResNetV2 Model loaded.')

# 连接两个预训练模型
# x = concatenate([dense_x, inception_res_x])

# 添加一个全连接层
x = Dense(2048, activation='relu')(dense_x)
x = Dropout(0.5)(x)
# 添加一个分类器，有两个类
predictions = Dense(1, activation='sigmoid')(x)

# 构建我们需要训练的完整模型
model = Model(inputs=main_input, outputs=predictions)

# 首先，我们只训练顶部的几层（随机初始化的层）
# 锁住所有 VGG 的卷积层
for layer in dense_model.layers:
    layer.trainable = False
# 锁住所有 InceptionV3 的卷积层
# for layer in inception_res_model.layers:
#     layer.trainable = False

# 编译模型（一定要在锁层之后进行编译）
model.compile(loss='binary_crossentropy',
              optimizer=SGD(lr=0.001, momentum=0.9, nesterov=True),
              metrics=['accuracy'])

# 画出网络模型的结构图
plot_model(model, to_file='model.png')

# 调用回调函数将日志信息写入TensorBoard 以动态的观察训练和测试指标的图像以及不同层的激活值直方图。
tb_path = os.path.join(ROOT_DIR, './logs')
tb = callbacks.TensorBoard(log_dir=tb_path)

# 在新的数据集上训练几轮
model.fit_generator(train_generator,
                    steps_per_epoch=1000,
                    epochs=EPOCHS,
                    callbacks=[tb],
                    validation_data=val_generator,
                    validation_steps=100)

# 使用模型跑测试集
gen = ImageDataGenerator()
test_gen = test_data_gen.flow_from_directory(test_data_path,
                                             target_size=(img_height, img_width),
                                             batch_size=BATCH_SIZE,
                                             shuffle=False,
                                             class_mode=None)
y_pred = model.predict_generator(test_gen)

# 将结果写入 .CSV 文件中
csv_path = os.path.join(ROOT_DIR, 'sampleSubmission.csv')
df = pd.read_csv(csv_path)
for i, f_name in enumerate(test_gen.filenames):
    idx = int(f_name[f_name.rfind('/')+1:f_name.rfind('.')])
    df.set_value(idx-1, 'label', y_pred[i])

df.to_csv('pred.csv', index=None)
df.head(10)
