import json 
from tqdm import tqdm 
import torch 
import torchmetrics

import pandas as pd 

from config import DefaultConfig
from data.face_dataset_crossattention import EmotionDataSet
from model.coatnetcrossattention import CoAtNet

# 定义模型文件夹
#test_path = '/home/yolo/network-model/train_logs/2023_May_30_PM_03_18_52_coatnet_CCTT_4_crossattention'
test_path = '/home/yolo/network-model/train_logs/2023_Jun_02_PM_11_30_11_coatnet_CCCC_4_crossattention'

def test_log_folder(log_path): 
    config = DefaultConfig()

    model = None 
    
    # # CoAtNet 中每个阶段的块数
    num_blocks = [2, 4, 8, 14, 4]
    # CoAtNet 中每个阶段的通道数
    channels = [32, 64, 128, 256, 1024]
    block_types=['C', 'C', 'C', 'C'] # CCCC, CCTT, TTTT 

    # 定义了输入图片的大小为 96x96
    image_size = 96
    # 实例化了一个 CoAtNet 模型，并将其输入大小设置为 96x96，通道数设置为 3
    model = CoAtNet((image_size, image_size), 4, num_blocks, channels, block_types=block_types, num_classes=7)
        

    test_set = EmotionDataSet('test', config.path.json_path, config.path.processed_data_folder)
    # 构建训练数据集的 DataLoader
    test_loader = torch.utils.data.DataLoader(
        # 用于数据集读取和处理的Dataset对象
        test_set, 
        # 每个 batch 的大小
        # 3 TTTT 256   CCCC   256   CCTT 256
        # 4 TTTT 256   CCCC   256   CCTT 256
        # resnet 256
        # InceptionV3_3 128
        # crossattention 8
        batch_size = 8,
        # 是否打乱数据
        shuffle=True, 
        # 多进程加载数据的进程数
        num_workers=1, 
        # 是否将数据存储于固定位置的显存中，若为 True，
        # 数据会被分配到固定位置的显存中，避免不必要的 CPU-GPU 数据传输，加速训练
        pin_memory=True, 
        # 如果最后一个 batch 的数据量不足 batch_size，是否舍去
        drop_last=True, 
        # 预取数据的因子，数据加载器将会启动 num_workers * prefetch_factor 个线程预取数据，以提高数据加载速度
        prefetch_factor=1, 
        # 是否使用持续的 workers
        persistent_workers=True 
    )

    # 读取训练日志文件夹中的 record.csv 文件，并存储在名为 table 的 DataFrame 中
    table = pd.read_csv(f'{log_path}/record.csv')
    # 按照 DICE.1 列进行降序排序，找到具有最高 DICE 分数的行，并获取其对应的 epoch 值
    best_epoch = table.sort_values(by=['DICE.1'], ascending=False)['epoch'].tolist()[0] 

    # 使用 torch.load 函数加载训练日志文件夹中的模型权重文件 
    state_dict = torch.load(f'{log_path}/models/epoch_{best_epoch}.pth')
    #print(state_dict['model'].keys())
    # 将加载的模型权重应用到先前定义的模型中
    model.load_state_dict(state_dict['model'])
    # 将模型设置为评估模式
    model.eval()
    # 将模型移动到 GPU 上进行加速
    model = model.cuda() 

    # 创建一个 Accuracy 的评估指标对象 acc，并指定评估任务为多分类 (task='multiclass')，
    # 类别数为 7 (num_classes=7)，计算平均值的方法为宏平均 (average='macro')
    acc = torchmetrics.Accuracy(task='multiclass', num_classes=7, average='macro').to('cuda')
    # 创建一个 Dice 的评估指标对象 dice，并指定类别数为 7 (num_classes=7)，
    # 计算平均值的方法为宏平均 (average='macro')
    dice =  torchmetrics.Dice(num_classes=7, average='macro').to('cuda')

    # 创建一个空列表 test_result，用于存储每个批次的评估结果
    test_result = []
    # 使用 torch.no_grad() 上下文管理器，禁用梯度计算
    with torch.no_grad():
        for *feature, target in tqdm(test_loader,ncols=80):
            # 将输入数据 feature 和目标标签 target 移动到 GPU 上，
            # 同时使用半精度浮点数进行计算（.cuda().half()）
            #feature, target = feature.cuda().half(), target.cuda()
            feature = [f.cuda().half() for f in feature]
            target = target.cuda()
            # 使用 torch.cuda.amp.autocast() 上下文管理器，启用混合精度计算
            with torch.cuda.amp.autocast():
                # 使用模型 model 对输入数据进行前向传播，并选择前7个输出维度（model(feature)[:,:7]）
                #print(model(feature))
                #print(model(feature).shape)
                prediction = model(feature)[:,:7] # [:, :7]
                #prediction = model(feature)
                # 使用评估指标 acc 和 dice 对预测结果和目标标签进行计算
                batch_acc = acc(prediction, target)
                batch_dice = dice(prediction, target)
            # 将每个批次的准确率和 Dice 分数添加到 test_result 列表中
            test_result.append([batch_acc, batch_dice])
    # 计算 test_result 列表中准确率和 Dice 分数的平均值，并将最佳模型的 epoch 值作为首个元素添加到结果列表中
    test_result = [best_epoch] + torch.tensor(test_result).mean(axis=0).tolist()
    # 使用 json.dump() 将结果列表以 JSON 格式保存到 test_result.json 文件中
    json.dump(
        {key: value for key, value in zip(["epoch", "Accuracy", 'F1'], test_result)},
        open(f"{log_path}/test_result.json", "w"),
        indent=4,
    )


if __name__ == '__main__':
    test_log_folder(test_path)