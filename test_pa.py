import tensorflow as tf
import simplejson
# import matplotlib.pyplot as plt
# %matplotlib inline
import threading
import tensorflow.contrib.slim as slim
from utils import data_utils, train_utils
import datetime
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2
import train
import pandas as pd
from shapely import wkt
import time
import sys
from inference import pred_for_each_quarter, test_input, stitch_mask

hypes = './hypes/hypes.json'
with open(hypes, 'r') as f:
    H = simplejson.load(f)
    H['batch_size'] = 1
    H['pad'] = 100
    H['x_width'] = 1920
    H['x_height'] = 1920
    H['print_iter'] = 100
    H['save_iter'] = 500
    H['crop_size'] = [1700, 1700]

    print_iter = H['print_iter']
    num_channel = H['num_channel']
    x_width = H['x_width']
    x_height = H['x_height']
    batch_size = H['batch_size']
    class_type = H['class_type']
    pad = H['pad']
    class_type = H['class_type']
    log_dir = H['log_dir']
    save_iter = H['save_iter']

img_in = tf.placeholder(dtype=tf.float32, shape=[batch_size, x_width, x_height, 16])
logits, pred = train.build_pred(img_in, H, 'test')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
saver = tf.train.Saver()
sess = tf.Session(config=config)
# saver.restore(sess, save_path='log_dir/ckpt_new/ckpt-12000')
# saver.restore(sess, save_path='log_dir/3-31_15-26_combo-jaccard/ckpt/ckpt-9000') # CLASS 0
# saver.restore(sess, save_path='log_dir/4-23_23-16_combo-jaccard/ckpt/ckpt-9000')  # CLASS 2
saver.restore(sess, save_path='log_dir/4-24_8-21_combo-jaccard/ckpt/ckpt-9000')  # CLASS 5

ids_with_instance = train_utils.generate_train_ids(class_type)   # 传递数据
print('IDs of training data with instance of class {} ({}): {}'.format(
    class_type, data_utils.CLASSES[class_type + 1], ids_with_instance))

jaccard_indices = {}

'''
for img_id in ids_with_instance:

    img_data = data_utils.ImageData(img_id)
    img_data.load_image()
    img_data.create_train_feature()
    img_data.create_label()

    mask_stack, shape_stack = pred_for_each_quarter(sess, img_in, pred, img_data, H)
    mask = stitch_mask(mask_stack, img_data.image_size, shape_stack, H)
    polygons = data_utils.mask_to_polygons(mask=mask, img_id=img_id, test=False, epsilon=1)
    true_polygons = data_utils.get_polygon_list(
        image_id=data_utils.train_IDs_dict[img_id], class_type=class_type + 1)

    jaccard_indices[data_utils.train_IDs_dict[img_id]] = \
        polygons.intersection(true_polygons).area / polygons.union(true_polygons).area

    alpha = 0.4
    fig, axs = plt.subplots(2, 2, figsize=[20, 20])
    print
    ('Processing ImageId: {} (No. {}); Class ({}): {}'.format(
        img_id, data_utils.train_IDs_dict[img_id], class_type, data_utils.CLASSES[class_type + 1]))

    fig.suptitle('Image (No. {}) Id {}; Class ({}): {}'.format(
        img_id, data_utils.train_IDs_dict[img_id], class_type, data_utils.CLASSES[class_type + 1]),
        fontsize=16)

    for i in range(2):
        for j in range(2):
            axs[0, 0].imshow(img_data.label[:, :, class_type], cmap=plt.cm.gray)
            axs[0, 0].set_title('True label for image: {}, class: {}'.format(
                data_utils.train_IDs_dict[img_id], data_utils.CLASSES[class_type + 1]))
            axs[0, 1].imshow(data_utils.scale_percentile(img_data.three_band_image))
            axs[0, 1].imshow(img_data.label[:, :, class_type], cmap=plt.cm.gray, alpha=alpha)
            axs[0, 1].set_title('3-band image with true label for image: {}, class: {}'.format(
                data_utils.train_IDs_dict[img_id], data_utils.CLASSES[class_type + 1]))
            axs[1, 0].imshow(mask, cmap=plt.cm.gray)
            axs[1, 0].set_title('Predicted label for image: {}, class: {}'.format(
                data_utils.train_IDs_dict[img_id], data_utils.CLASSES[class_type + 1]))
            axs[1, 1].imshow(data_utils.scale_percentile(img_data.three_band_image))
            axs[1, 1].imshow(mask, cmap=plt.cm.gray, alpha=alpha)
            axs[1, 1].set_title('3-band image with predicted label for image: {}, class: {}'.format(
                data_utils.train_IDs_dict[img_id], data_utils.CLASSES[class_type + 1]))
            fig1 = plt.gcf()
            # plt.show()  # 试试能否显示图片/
            fig1.savefig("/home/administrator/桌面/dstl_unet-master/pic_save/tt%d_%d_%d" % (img_id, i, j))  # 用id来辨别图像
        # fig = plt.gcf()
        # plt.show()  # 试试能否显示图片
        # fig.savefig("/home/administrator/桌面/dstl_unet-master/pic_save/sub_tt%d" % i)
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    # if img_id ==7: break

print
(('Jaccard indices {}'.format(jaccard_indices)))
# print
# (('Mean Jaccard index {}'.format(np.mean(jaccard_indices.values()))))

'''
# ids_w_o_instance = sorted(list(set(range(25)) - set(ids_with_instance)))
ids_w_o_instance = sorted(list(set(range(16)) - set(ids_with_instance)))

print
('IDs of training data w/o instance of class {} ({}): {}'.format(
    class_type, data_utils.CLASSES[class_type + 1], ids_w_o_instance))

for img_id in ids_w_o_instance:
    print
    ('Processing ImageId (No. {}): {}; Class ({}): {}'.format(
        img_id, data_utils.train_IDs_dict[img_id], class_type, data_utils.CLASSES[class_type + 1]))

    img_data = data_utils.ImageData(img_id)
    img_data.load_image()
    img_data.create_train_feature()

    mask_stack, shape_stack = pred_for_each_quarter(sess, img_in, pred, img_data, H)
    mask = stitch_mask(mask_stack, img_data.image_size, shape_stack, H)
    polygons = data_utils.mask_to_polygons(mask=mask, img_id=img_id, test=False, epsilon=1)
    true_polygons = data_utils.get_polygon_list(
        image_id=data_utils.train_IDs_dict[img_id], class_type=class_type + 1)

    jaccard_indices[data_utils.train_IDs_dict[img_id]] = \
        polygons.intersection(true_polygons).area / polygons.union(true_polygons).area \
            if polygons.union(true_polygons).area else 1.
    # plt.show()  # 试试能否显示图片

print("no problem1!!!")

print
('Jaccard indices {}'.format(jaccard_indices))
# print
# ('Mean Jaccard index {}'.format(np.mean(jaccard_indices.values())))

print("no problem2!!!")

for img_id in range(30, 35):

    img_data = data_utils.ImageData(img_id, phase='test')
    img_data.load_image()
    img_data.create_train_feature()

    mask_stack, shape_stack = pred_for_each_quarter(sess, img_in, pred, img_data, H)
    mask = stitch_mask(mask_stack, img_data.image_size, shape_stack, H)

    alpha = 0.4
    fig, axs = plt.subplots(1, 2, figsize=[20, 10])
    print
    ('Processing ImageId: {} (No. {}); Class ({}): {}'.format(
        img_id, data_utils.test_IDs_dict[img_id], class_type, data_utils.CLASSES[class_type + 1]))

    fig.suptitle('Image (No. {}) Id {}; Class ({}): {}'.format(
        img_id, data_utils.test_IDs_dict[img_id], class_type, data_utils.CLASSES[class_type + 1]),
        fontsize=16)

    print("no problem!!!")

    for i in range(2):
        for j in range(2):
            axs[0].imshow(mask, cmap=plt.cm.gray)
            axs[0].set_title('Predicted label for image: {}, class: {}'.format(
                data_utils.test_IDs_dict[img_id], data_utils.CLASSES[class_type + 1]))
            axs[1].imshow(data_utils.scale_percentile(img_data.three_band_image))
            axs[1].imshow(mask, cmap=plt.cm.gray, alpha=alpha)
            axs[1].set_title('3-band image with predicted label for image: {}, class: {}'.format(
                data_utils.test_IDs_dict[img_id], data_utils.CLASSES[class_type + 1]))
            plt.savefig("/home/administrator/桌面/dstl_unet-master/pic_save/t_another%d_%d_%d" % (img_id, i, j))
            # plt.show()  # 试试能否显示图片
    # plt.show()  # 试试能否显示图片
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
