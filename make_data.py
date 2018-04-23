#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-3-07 17:35:39
# @Author  : Mei
from __future__ import print_function
import os
import shutil
import time

total_time = time.time()
start_time = time.time()
ROOT_DIR = os.getcwd()
train_path = os.path.join(ROOT_DIR, 'train')

'''
# os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。
# 这个列表以字母顺序。 它不包括 '.' 和'..' 即使它在文件夹中。
'''
train_data = os.listdir(train_path)
print("数据读取完毕. %.2fs" % (time.time() - start_time))
'''
# filter(function, iterable) 函数用于过滤序列
# 过滤掉不符合 function 条件的元素，返回由符合条件元素组成的新列表。
'''
start_time = time.time()
train_cat = filter(lambda x: x[:][:3] == 'cat', train_data)
train_dog = filter(lambda x: x[:][:3] == 'dog', train_data)
print("数据分类完毕. %.2fs" % (time.time() - start_time))


def rmrf_mkdir(dir_name):
    if os.path.exists(dir_name):
        # 删除已存在的文件夹
        print("存在该名称的文件夹，删除中...")
        shutil.rmtree(dir_name)
    os.mkdir(dir_name)


# 创建训练集文件夹
start_time = time.time()
print("创建文件夹...")
new_train_path = os.path.join(ROOT_DIR, 'train_data')
rmrf_mkdir(new_train_path)
new_train_path_cat = os.path.join(new_train_path, 'cat')
os.mkdir(new_train_path_cat)
new_train_path_dog = os.path.join(new_train_path, 'dog')
os.mkdir(new_train_path_dog)
print("文件夹创建完毕. %.2fs" % (time.time() - start_time))

# 创建验证集文件夹
start_time = time.time()
print("创建文件夹...")
new_validation_path = os.path.join(ROOT_DIR, 'validation_data')
rmrf_mkdir(new_validation_path)
new_validation_path_cat = os.path.join(new_validation_path, 'cat')
os.mkdir(new_validation_path_cat)
new_validation_path_dog = os.path.join(new_validation_path, 'dog')
os.mkdir(new_validation_path_dog)
print("文件夹创建完毕. %.2fs" % (time.time() - start_time))

start_time = time.time()
print("给分类好的数据创建软连接（替身）到对应的文件夹中...")
# 训练集
for file_name in train_cat[:-1250]:
    os.symlink(train_path + '/' + file_name, new_train_path_cat + '/' + file_name)

for file_name in train_dog[:-1250]:
    os.symlink(train_path + '/' + file_name, new_train_path_dog + '/' + file_name)

# 验证集
for file_name in train_cat[-1250:]:
    os.symlink(train_path + '/' + file_name, new_validation_path_cat + '/' + file_name)

for file_name in train_dog[-1250:]:
    os.symlink(train_path + '/' + file_name, new_validation_path_dog + '/' + file_name)

print("创建软连接成功. %.2fs" % (time.time() - start_time))
print("总用时: %.2fs" % (time.time() - total_time))
