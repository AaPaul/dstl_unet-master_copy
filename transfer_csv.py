import csv
import sys
import numpy as np
import pprint as pp
from PIL import Image
import matplotlib.pyplot as plt
import time
import os

image_number = [0, 0, 0, 0, 0, 0, 0]
couter_num = 1
ccsv = 'E:\\workspace\\mystoreroom\\dstl_unet-master\\submission\\class_0.csv'
with open(ccsv, 'r') as csvfile:
    pre_reader = csv.DictReader(csvfile)  # 直接生成一个pre_reader，用于迭代读出
    for reader in pre_reader:  # 迭代读出pre_reader里的数据
        x = reader["MultipolygonWKT"]  # 取出其中的像素点
        x = x.split()  # split为分裂，x原来是一个字符串，用这个split后就分裂成字符串数组
        #        print(x)

        arry = [float(i) for i in x]  # 将x中一个个字符转化为int类型，arry是个int数组
        #        print(arry)
        #        print(type(arry[1]))

        result = np.array(arry)  # 生成np 的array
        result = result.reshape([48, 48])  # 变成48*48的形状
        #        pp.pprint(result)#显示一些转化后的结果
        image = Image.fromarray(result)  # 从数据，生成image对象
        image = image.resize([384, 384], Image.ANTIALIAS)  # 抗锯齿的放大

        plt.figure("test")  #
        plt.imshow(image)  # 建立窗口
        plt.show()  # 显示图片
        time.sleep(3)  # 延时300ms

        #        image.show()#显示图片
        os.system("pause")
        number = reader["ImageId"]  # 读出这张图片对应的表情标签
        #    print(number)
        name = "/image" + str(image_number[int(number)]) + ".png"
        path = os.path.join("./picture/", number)  # 生成这个表情图片对应的路径
        #    print((path+name))
        print("正在保存%s类型表情%s" % (number, name))
        image_save = image.convert("L")  # 转化为灰度图片(这一步很关键，不然图片不能以png格式保存)
        image_save.save(path + name)  #
        print("第%d保存成功！" % (couter_num))  # 显示已经保存了第几张
        image_number[int(number)] = image_number[int(number)] + 1
        couter_num += 1