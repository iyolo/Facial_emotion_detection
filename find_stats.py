import os 
import json 
from PIL import Image 

import numpy as np 


def main():
    # 指定图像数据集路径
    path = '/home/yolo/network-model/organized'
    # 统计所有图像像素值的和，以便计算图像通道的平均值
    pixels_count = 0 
    # 统计所有图像各个通道像素值的和，以便计算图像通道的平均值
    channel_sum = np.array([0, 0, 0])

    # 遍历所有图像，计算图像通道的平均值
    for each_class in os.listdir(path):
        class_dir = f'{path}/{each_class}'
        for each_image in os.listdir(class_dir): 
            image = np.array(Image.open(f'{class_dir}/{each_image}'))
            channel_sum += image.sum(axis=(0, 1))
            pixels_count += image.shape[0] * image.shape[1] 
    
    channel_mean = channel_sum / pixels_count

    # 统计所有图像各个通道像素值的平方和，以便计算图像通道的标准差
    channel_std = np.array([0, 0, 0], dtype=np.float64)

    # 遍历所有图像，计算图像通道的标准差
    for each_class in os.listdir(path):
        class_dir = f'{path}/{each_class}'
        for each_image in os.listdir(class_dir): 
            image = np.array(Image.open(f'{class_dir}/{each_image}'))
            # 对图像各个通道像素值除以该通道的平均值，然后取平方，以便计算该通道的标准差
            image = (image / channel_mean) ** 2 
            channel_std += image.sum(axis=(0, 1))
    channel_std = np.sqrt(channel_std)

    # 将图像通道的平均值和标准差存储到 JSON 文件中
    json.dump({'mean': channel_mean.tolist(), 'std': channel_std.tolist()}, open(f'{path}/stats.json'))


if __name__ == '__main__':
    main() 