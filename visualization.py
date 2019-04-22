import matplotlib
import matplotlib.pyplot as plt
from utils import data_utils, train_utils
# %matplotlib inline
import cv2
import numpy as np
from matplotlib import cm
from PIL import Image
from descartes.patch import PolygonPatch
from shapely import wkt
from shapely import affinity
import matplotlib



new_style = {'grid': False}
matplotlib.rc('axes', **new_style)

train_input = train_utils.input_data(class_id=0, crop_size=144, crop_per_img=1, rotation=360, verbose=True, train=True)
# img, label = train_input.next()
img, label = next(train_input)

print(11111111111)
print (img.shape, label.shape)
print (123123123123123)


ind = 0
# fig, axs = plt.subplots(5,5, figsize=[20,20])
# 这里我改成4*4，因为删除了一些数据，没有25个了，只有19个，展示前16个
fig, axs = plt.subplots(4,4, figsize=[20,20])
for i in range(0,4):
    for j in range(0,4):
        # axs[i,j].imshow(data_utils.scale_percentile(img[ind, :,:,9:12]))
        # figAll = data_utils.scale_percentile(img[ind, :,:,9:12])
        # figAll[ind].savefig("picture%d"%ind,format='png')
        axs[i,j].imshow(data_utils.scale_percentile(img[ind, :,:,9:12]))  # 这里是表示3通道，选择第三通道的9到12像素？   https://www.jianshu.com/p/f2e88197e81d
        ind = ind+1
plt.savefig("test0.png")
plt.show()


ind = 0
fig, axs = plt.subplots(5,5, figsize=[20,20])
for i in range(4):
    for j in range(4):
        axs[i,j].imshow(label[ind, :, :], cmap=plt.cm.gray)
        # plt.show(axs[i,j])
        ind = ind+1
plt.savefig("test00.png")
plt.show()



data_utils.CLASSES
img_data = data_utils.ImageData(17)
# load data
img_data.load_image()
img_data.create_label()
img_data.create_train_feature()
img_data.visualize_image(plot_all=True)
plt.savefig("test_all0.png")
plt.show()
img_data.visualize_label()


data_utils.plot_bar_stats()

data_utils.plot_stats(title='Total area', value='TotalArea')

data_utils.plot_stats(title='Counts', value='Counts')