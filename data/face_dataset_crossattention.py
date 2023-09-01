import json 

from PIL import Image 

import numpy as np 

import torch 

from openpose import pyopenpose as op


from data.trivial_aug import TrivialAugment

import pickle

# 定义了图像数据集的平均值和标准差，用于对数据进行标准化处理,
# 将原数组的第一维改为 3，第二维改为 1，第三维改为 1，也就是将一个一维的数组（长度为 3）变为了一个三维数组
channel_std = np.array([67.67195579, 61.77343565, 61.37423929]).reshape(3, 1, 1).astype(np.float32)
channel_mean = np.array([146.67694409, 114.62696945, 102.31048249]).reshape(3, 1, 1).astype(np.float32)


# 继承了 PyTorch 的 Dataset 类的类，用于加载表情识别数据集
class EmotionDataSet(torch.utils.data.Dataset): 

    def __init__(self, mode, json_path, path): 
        # 根据指定的 mode 和 json_path 加载文件名列表,mode 参数是数据集的模式，可以是 'train'、'val' 或 'test'
        self.files = json.load(open(json_path))[mode]
        # 指定数据集所在路径
        self.path = path
        # 数据增强类实例
        self.augmentor = TrivialAugment()
        # 记录当前数据集的模式，包括 train、val、test
        self.mode = mode 
        # 初始化图像像素值的平均值和标准差
        self.mean = channel_mean
        self.std = channel_std
        # 计算数据增强因子,对验证或测试数据进行轻微的扩充
        self.factor = 10 if mode == 'train' else 1 
           # 加载保存的数据
        with open('data.pkl', 'rb') as f:
            self.loaded_data = pickle.load(f)
        

    # 定义了数据集的长度，即包含多少个样本
    def __len__(self):
        return len(self.files) * self.factor
        
    # 获取增强后的图片
    def __getitem__(self, index):
        # 由于训练集是经过扩增的，因此在训练集中需要使用整数除法 // 除以 self.factor 来获得真实的文件名
        filename = self.files[index//self.factor]
        # 从文件路径中获取类别 class_id
        class_id = int(filename.split('/')[0])
        # 使用 Image.open 方法读取图像文件
        file = Image.open(f'{self.path}/{filename}') # RGB

        #使用Coatnet model

        # # 不使用openpose
        # # augment 
        # # 根据模式 self.mode 来决定是否进行数据增强 如果模式为 'train'，则对图像进行增强处理，
        # # 否则不做任何处理，直接返回读取的图像。
        # if self.mode == 'train': 
        #     file = self.augmentor(file)
        # # 将PIL.Image格式的图像转换为numpy.ndarray格式表示将原数组的第一个维度（颜色通道）移动到最后，
        # # 将原数组的第二个维度（高度）移到第一个位置，将原数组的第三个维度（宽度）移到第二个位置。
        # ndarray = np.array(file).transpose(2, 0, 1)
        # # 对图像数据进行归一化操作，即对每个像素值减去该通道的平均值，再除以该通道的标准差，
        # # 以使得所有像素值的分布范围在0到1之间。这样做的目的是使得图像数据分布更加合理，方便训练
        # ndarray = (ndarray - self.mean) / self.std
        # # return input_data, target_data
        # # 将 ndarray 转换为 PyTorch 的 Tensor 格式，并且将通道数缩减为 96 以适应模型的输入尺寸。
        # # 然后将类别 ID 转换为 PyTorch 的 LongTensor 格式，并返回模型需要的两个输入：图像数据和标签
        # #print(torch.from_numpy(ndarray)[:,:96,:96].shape)
        # return torch.from_numpy(ndarray)[:,:96,:96], torch.tensor(class_id).long()


        # 使用openpose   
        # 获取数据中的openposedata
        keypoint_channel = self.loaded_data[filename].toarray()
        # augment 
        # 根据模式 self.mode 来决定是否进行数据增强 如果模式为 'train'，则对图像进行增强处理，
        # 否则不做任何处理，直接返回读取的图像。
        if self.mode == 'train': 
            file, keypoint_channel = self.augmentor(file, keypoint_channel)
        # 将PIL.Image格式的图像转换为numpy.ndarray格式表示将原数组的第一个维度（颜色通道）移动到最后，
        # 将原数组的第二个维度（高度）移到第一个位置，将原数组的第三个维度（宽度）移到第二个位置。
        ndarray = np.array(file).transpose(2, 0, 1)
        # 对图像数据进行归一化操作，即对每个像素值减去该通道的平均值，再除以该通道的标准差，
        # 以使得所有像素值的分布范围在0到1之间。这样做的目的是使得图像数据分布更加合理，方便训练
        ndarray = (ndarray - self.mean) / self.std
        # 添加openpose通道
        # ndarray = np.concatenate([ndarray, np.expand_dims(keypoint_channel, axis=0)], axis=0)
        p1 = torch.from_numpy(ndarray)[:, :96, :96].half()
        p2 = torch.from_numpy(keypoint_channel).unsqueeze(0)[:, :96 , :96].half()

        # return input_data, target_data
        # 将 ndarray 转换为 PyTorch 的 Tensor 格式，并且将通道数缩减为 96 以适应模型的输入尺寸。
        # 然后将类别 ID 转换为 PyTorch 的 LongTensor 格式，并返回模型需要的两个输入：图像数据和标签
        return p1, p2, torch.tensor(class_id).long()


        # ## 使用ResNet model
        # # augment 
        # # 根据模式 self.mode 来决定是否进行数据增强 如果模式为 'train'，则对图像进行增强处理，
        # # 否则不做任何处理，直接返回读取的图像。
        # if self.mode == 'train': 
        #     file = self.augmentor(file)
        # # 将PIL.Image格式的图像转换为numpy.ndarray格式表示将原数组的第一个维度（颜色通道）移动到最后，
        # # 将原数组的第二个维度（高度）移到第一个位置，将原数组的第三个维度（宽度）移到第二个位置。
        # ndarray = np.array(file).transpose(2, 0, 1)
        # # 对图像数据进行归一化操作，即对每个像素值减去该通道的平均值，再除以该通道的标准差，
        # # 以使得所有像素值的分布范围在0到1之间。这样做的目的是使得图像数据分布更加合理，方便训练
        # ndarray = (ndarray - self.mean) / self.std
        # # print(torch.from_numpy(ndarray)[:,:96,:96].type())
        # return torch.from_numpy(ndarray)[:,:96,:96], torch.tensor(class_id).long()


        # ## 使用inception_v3 model
        # from torchvision import transforms
        # # 定义一个transform
        # transform = transforms.Compose([
        #     # Resize images to 299x299
        #     transforms.Resize((299, 299),interpolation=transforms.InterpolationMode.BICUBIC),
        #     # Convert images to PyTorch tensor
        #     transforms.ToTensor(),  
        #     # Normalize images
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
        # ])
        # # 创建一个只包含Resize操作的transform
        # resize = transforms.Resize((299, 299))
        # # 对图像进行尺寸调整：
        # file_resized = resize(file)
        # #print(type(file))
        # # print(type(file_resized))
        # if self.mode == 'train': 
        #     file_resized = self.augmentor(file_resized)
        # # 将PIL.Image格式的图像转换为numpy.ndarray格式表示将原数组的第一个维度（颜色通道）移动到最后，
        # # 将原数组的第二个维度（高度）移到第一个位置，将原数组的第三个维度（宽度）移到第二个位置。
        # ndarray = np.array(file_resized).transpose(2, 0, 1)
        # # 对图像数据进行归一化操作，即对每个像素值减去该通道的平均值，再除以该通道的标准差，
        # # 以使得所有像素值的分布范围在0到1之间。这样做的目的是使得图像数据分布更加合理，方便训练
        # ndarray = (ndarray - self.mean) / self.std
        # return torch.from_numpy(ndarray), torch.tensor(class_id).long()
        
    