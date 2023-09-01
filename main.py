import os
import re 
import pickle 
import random
import shutil
from datetime import datetime

import torch
import numpy as np

from accelerate import Accelerator

from config import DefaultConfig

# pipreqs . 

# https://image-net.org/data/winter21_whole.tar.gz

# 过滤给定项目列表中的特定类型的文件和文件夹
def filter_files(path, items):
    # 创建一个空集合，用于存储要忽略的文件
    ignore_files = set()
    for each_item in items:
        # 如果项目是以 ".py" 结尾的文件，则跳过
        if each_item.endswith(".py"):
            pass
        # 如果项目是以 ".md" 结尾的文件，则跳过
        elif each_item.endswith(".md"):
            pass
        # 如果项目以 "." 开头，则将其添加到忽略文件集合中
        elif each_item.startswith("."):
            ignore_files.add(each_item)
        # 如果项目以 "_" 开头，则将其添加到忽略文件集合中
        elif each_item.startswith("_"):
            ignore_files.add(each_item)
        # 如果项目是一个目录，则跳过
        elif os.path.isdir(f"{path}/{each_item}"):
            pass
        # 将其他类型的项目添加到忽略文件集合中
        else:
            ignore_files.add(each_item)
    return ignore_files


def main():
    # 实例化默认配置类
    config = DefaultConfig()
    # 实例化加速器
    accelerator = Accelerator(
        # 是否拆分批次，使其适应多个设备
        split_batches=True,
        # 是否在优化器更新时调用步骤调度器
        step_scheduler_with_optimizer=True, 
        # 梯度累积的步数
        gradient_accumulation_steps = config.train.gradient_accumulate_steps
    )
 
    # setup logs
    # 获取当前时间并格式化为字符串
    time = datetime.now().strftime("%Y_%b_%d_%p_%I_%M_%S")
    # 创建日志路径
    log_path = f"{config.path.train_log_folder}/{time}_{config.train.case_name}"
    # 将日志路径保存到配置中
    config.path.log_path = log_path
    
    # 创建分隔符字符串
    sep = '-' * config.report.line_width
    # 构建报告字符串
    report_str = f'initialized {accelerator.num_processes} processes for DDP\n'
    report_str += f'using device: {accelerator.device}; using amp mode: {accelerator.mixed_precision}\n'
    report_str += f'syncing gradient every: {config.train.gradient_accumulate_steps} steps'
    # 打印报告字符串
    print(f'\n{sep}\n{report_str}\n{sep}\n')
    
    # 获取候选日志文件夹
    candidates = os.listdir(config.path.train_log_folder) 
     # 创建时间正则表达式
    time_regex = re.compile('\d{4}_\w+_\d{2}_\w{2}_\d{2}_\d{2}_\d{2}')

    # 导入训练器类
    from trainer.basic_supervised_trainer import SupervisedNetworkTrainer
    #from trainer.basic_supervised_trainer_crossattention import SupervisedNetworkTrainer
     # 初始化训练器
    trainer = SupervisedNetworkTrainer(config, accelerator)
    # 根据配置选择是否从检查点恢复训练
    if config.train.restore == 'none': 
        trainer.init() 
    elif config.train.restore == 'latest':
        latest = sorted(os.listdir(config.path.train_log_folder), 
                         key=lambda each: datetime.strptime(time_regex.search(each).group(0), 
                                                            "%Y_%b_%d_%p_%I_%M_%S"))[-1]
        trainer.load_checkpoint(f'{config.path.train_log_folder}/{latest}/checkpoint.pth')
    elif config.train.restore in candidates: 
        trainer.load_checkpoint(f'{config.path.train_log_folder}/{config.train.restore}/checkpoint.pth')
    else:
        raise ValueError

    # 如果是主进程，则执行以下操作
    if accelerator.is_main_process:
        # 复制当前目录中的代码到日志路径
        shutil.copytree(os.getcwd(), f"{log_path}/code_used", ignore=filter_files)
        # 创建模型文件夹
        os.mkdir(f"{log_path}/models")
        trainer.describe_trainer()

    # 运行训练
    trainer.run_training()


if __name__ == "__main__":
    # 实例化默认配置类
    config = DefaultConfig()
    # 启用或禁用CuDNN后端
    torch.backends.cudnn.enable = config.cuda.cudnn
    # 根据网络输入启用或禁用CuDNN后端的自动调优
    torch.backends.cudnn.benchmark = config.cuda.benchmark
    # 为CuDNN后端启用或禁用确定性模式
    torch.backends.cudnn.deterministic = config.cuda.deterministic

     # 如果配置中提供了随机种子
    if config.cuda.seed is not None:
        # 检查种子为整数
        assert isinstance(config.cuda.seed, int), "seed must be int or None"
        # 设置Python内置random库的种子
        random.seed(config.cuda.seed)
        # 设置NumPy库的种子
        np.random.seed(config.cuda.seed)
        # 设置PyTorch库的种子
        torch.manual_seed(config.cuda.seed)

    main()
