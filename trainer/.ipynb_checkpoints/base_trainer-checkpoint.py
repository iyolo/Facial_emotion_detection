import os 
import re 
# 提供对象的序列化和反序列化
import pickle 

from datetime import datetime
# 提供创建抽象基类和抽象方法的功能
from abc import ABC, abstractmethod

# 提供进度条显示功能
from tqdm import tqdm

import torch 
import numpy as np 
import pandas as pd 


class NetworkTrainer(ABC): 
    # 初始化方法，接收配置信息和加速器对象
    def __init__(self, config, accelerator): 
        # 配置信息
        self.config = config 
        # 加速器对象
        self.accelerator = accelerator 

        # 日志路径
        self.log_path = config.path.log_path
        # 训练的总轮次
        self.total_epochs = config.train.total_epochs
        # 梯度累积步数
        self.accumulate_steps = config.train.gradient_accumulate_steps 

         # 报告行宽
        self.line_width = config.report.line_width

        # 初始化前一轮的验证集损失
        self.prev_val_loss = 100
        # 模型保存模式
        self.save_mode = config.train.save_mode
        # 确保保存模式是预定义的三种之一
        assert self.save_mode in ("best", "improve", "all")

        # 指标名称列表
        self.metric_names = ["epoch"]
        # 创建一个空的DataFrame，用于记录训练过程中的指标
        self.record_table = pd.DataFrame()

     # 抽象方法，初始化模型、优化器、学习率调度器等
    @abstractmethod
    def init(self):
        pass 

    # 定义模型的前向传播
    @abstractmethod
    def forward_model(self):
        pass 
    
    # 定义模型的反向传播
    @abstractmethod
    def backward_model(self): 
        pass 

    # 定义训练一个epoch的过程
    @abstractmethod
    def train_one_epoch(self):
        pass 

    # 定义在验证集上评估一个epoch的过程
    @abstractmethod
    def val_one_epoch(self):
        pass 

    # 运行训练过程的方法
    def run_training(self): 
        # 打印分隔线
        print("-" * self.line_width)
        # 遍历每一个训练轮次
        for current_epoch in range(1, (self.total_epochs + 1 )):
        #for current_epoch in range(1, 2):
            self.current_epoch = current_epoch 
            # start training
            # 记录开始训练的时间
            start_time = datetime.now()
            # 将模型设置为训练模式
            self.model.train()
            # 训练一个epoch并获取训练损失
            train_loss = self.train_one_epoch(f"[{current_epoch}/{self.total_epochs} epochs]")
            # 收集并平均每个设备的训练损失
            train_loss = self.accelerator.gather(train_loss).mean(dim=0).tolist() # ce, dice
            # 计算训练一个epoch所用的时间
            train_time = (datetime.now() - start_time).total_seconds()

            # start validation
            # 开始在验证集上评估模型
            start_time = datetime.now()
            # 将模型设置为评估模式
            self.model.eval()
            # 在验证集上评估一个epoch并获取验证损失
            val_losses = self.val_one_epoch()
            # 收集并平均每个设备的验证损失
            val_losses = self.accelerator.gather(val_losses).mean(dim=0).tolist()

            # record and print everything
            # 记录并打印训练过程中的信息
            if self.accelerator.is_main_process:
                # 如果保存模式为"best"且当前验证损失小于前一轮的验证损失
                if self.save_mode == "best" and sum(val_losses) > self.prev_val_loss:
                    # 保存模型
                    self.save_inference(f"{self.log_path}/model_best.pth")
                    # 更新前一轮的验证损失
                    self.prev_val_loss = sum(val_losses)
                elif self.save_mode == "improve" and sum(val_losses) > self.prev_val_loss:
                    self.save_inference(f"{self.log_path}/models/epoch_{current_epoch}.pth")
                    self.prev_val_loss = sum(val_losses)
                elif self.save_mode == "all":
                    self.save_inference(f"{self.log_path}/models/epoch_{current_epoch}.pth")

                # print([current_epoch, train_loss, *val_losses])
                # 创建当前轮次的记录
                epoch_record = [current_epoch, *train_loss, *val_losses]
                # 将记录转化为DataFrame
                epoch_table = pd.DataFrame(data=[epoch_record], columns=self.metric_names)
                 # 添加到总的记录表中
                self.record_table = pd.concat([self.record_table, epoch_table], axis=0)
                # 将记录表保存为csv文件
                self.record_table.to_csv(f"{self.log_path}/record.csv", index=False)

                # 计算在验证集上评估一个epoch所用的时间
                val_time = (datetime.now() - start_time).total_seconds()

                # 将训练时间转化为分钟和秒的形式
                train_time = f"train time: {int(train_time//60):0>2}:{int(train_time%60):0>2}"
                # 将验证时间转化为分钟和秒的形式
                val_time = f"val time: {int(val_time//60):0>2}:{int(val_time%60):0>2}"
                # 打印训练和验证的时间，以及当前epoch的指标
                print(f' {train_time}, {val_time}\n{epoch_table.to_string()}\n{"-" * self.line_width}')
            # 等待所有设备完成当前epoch的操作
            self.accelerator.wait_for_everyone()

    # 保存训练过程中的检查点，包括模型状态、优化器状态、学习率调度器状态等
    @abstractmethod
    def save_checkpoint(self):
        pass 
    
    # 加载训练过程中的检查点，以便于从中断的地方继续训练
    @abstractmethod
    def load_checkpoint(self):
        pass 

     # 加载训练过程中的检查点，以便于从中断的地方继续训练
    @abstractmethod
    def save_inference(self):
        pass 

    # 保存用于推理的模型，通常是训练完成后的模型
    @abstractmethod
    def load_inference(self): 
        pass 

    # 主进程包装器，用于包装只在主进程中运行的方法
    def main_process_wrapper(func):
        def inner(self, *args, **kwargs): 
             # 如果当前设备是主设备
            if self.accelerator.is_local_main_process:
                # 执行方法
                func(self, *args, **kwargs)
        return inner 
    
     # 保存基础的检查点，包括模型状态、优化器状态和学习率调度器状态
    def save_basic_checkpoint(self):
        # 获取模型的状态字典
        model = self.accelerator.unwrap_model(self.model).state_dict()
        # 获取优化器的状态字典
        optimizer = self.model_optim.state_dict() 
         # 获取学习率调度器
        scheduler = self.model_scheduler 
        # 将检查点保存为pickle文件
        pickle.dump({'model': model, 'optimizer': optimizer, 'scheduler': scheduler}, open(f'{self.log_path}/checkpoint.pth', 'wb'))
