import torch 

from tqdm import tqdm 

from .base_trainer import NetworkTrainer

from loss_func.segmentation import CE_DICE, ValLoss
from data.face_dataset_crossattention import EmotionDataSet
#from model.coatnet import CoAtNet
from model.coatnetcrossattention import CoAtNet
#from model.coatnetsource import CoAtNet

class SupervisedNetworkTrainer(NetworkTrainer):
    # 初始化时创建数据集、数据加载器、模型和优化器
    def init(self):
        # 构建训练数据集,实例化对象
        train_dataset = EmotionDataSet('train', self.config.path.json_path, self.config.path.processed_data_folder)
        # 构建训练数据集的 DataLoader
        train_loader = torch.utils.data.DataLoader(
            # 用于数据集读取和处理的Dataset对象
            train_dataset, 
            # 每个 batch 的大小
            batch_size = self.config.train.train_batch_size, 
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

        # 加载验证集,实例化对象
        val_dataset = EmotionDataSet('val', self.config.path.json_path, self.config.path.processed_data_folder)
        # 创建 PyTorch 的 DataLoader 对象，用于加载和处理数据集
        val_loader = torch.utils.data.DataLoader(
            # val_dataset对象
            val_dataset, 
            # 每个 batch 的大小
            batch_size = self.config.train.train_batch_size, 
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

        # 训练集数据的迭代器中可以迭代的次数，也就是训练集中有多少个batch
        total_batches = len(train_loader)
        # 每个 epoch 中需要进行多少个 update操作
        self.update_per_epoch = total_batches // self.config.train.gradient_accumulate_steps

        # 将原始的 PyTorch DataLoader 对象
        # 使用 PyTorch Lightning 的加速器（accelerator）进行封装，使得数据加载可以更加高效
        self.train_loader = self.accelerator.prepare_data_loader(train_loader)
        self.val_loader = self.accelerator.prepare_data_loader(val_loader)

        # 定义了训练和验证损失函数的实例对象
        self.train_loss = CE_DICE(self.config.data.num_classes, self.accelerator.device)
        self.val_loss = ValLoss(self.config.data.num_classes, self.accelerator.device)
        # 是一个字符串列表，它包含所有训练和验证指标的名称
        self.metric_names += self.train_loss.names + self.val_loss.names

        # CoAtNet 中每个阶段的块数
        num_blocks = [2, 4, 8, 14, 4]
        # CoAtNet 中每个阶段的通道数
        channels = [32, 64, 128, 256, 1024]
        # block_types=['C', 'C', 'C', 'C'] # CCCC, CCTT, TTTT 
        block_types = list(self.config.train.case_name.split('_')[1])

        # 定义了输入图片的大小为 96x96
        image_size = 96
        # 实例化了一个 CoAtNet 模型，并将其输入大小设置为 96x96，通道数设置为 3
        model = CoAtNet((image_size, image_size), 4, num_blocks, channels, block_types=block_types, num_classes=self.config.data.num_classes)
        
        import torchvision 
        #model = torchvision.models.resnet101(torchvision.models.ResNet101_Weights.IMAGENET1K_V2)
        #model = torchvision.models.resnet101()
        #model = torchvision.models.inception_v3(torchvision.models.Inception_V3_Weights)
        #model = torchvision.models.inception_v3()

        # 定义了一个 Adam 优化器
        optim = torch.optim.Adam(
            # 返回一个包含了model中所有需要学习的参数的迭代器
            model.parameters(), 
            # 学习率,模型参数沿着梯度下降的方向更新的大小
            lr=self.config.train.max_lr, 
            # L2 正则化系数，它有助于防止模型过度拟合训练数据
            weight_decay=1e-5
        )
        # OneCycleLR 学习率调整策略，
        # 它将学习率在训练过程中从一个较小值逐渐升高到一个较大值，然后再逐渐降低，最终回到一个较小的值
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            # 优化器，即需要进行学习率调整的优化器
            optim,
            # 学习率上限
            max_lr=self.config.train.max_lr, 
            # 训练总步数，即训练集中的样本数除以 batch size，再乘以总 epoch 数量
            total_steps=(self.update_per_epoch + 1) * self.config.train.total_epochs
        )

        # 使用 accelerator 对象的方法对模型、优化器和学习率调度器进行了准备
        self.model = self.accelerator.prepare_model(model)
        self.model_optim = self.accelerator.prepare_optimizer(optim)
        self.model_scheduler = self.accelerator.prepare_scheduler(scheduler)
    
    # 向前传播
    def forward_model(self, feature):
        return self.model(feature)

    # 后向传播
    def backward_model(self, prediction, target): 
        # 计算预测值与真实值 target 之间的损失 loss
        loss = self.train_loss(prediction, target)
        # 对多个损失 loss 求和，得到总的损失 total_loss
        total_loss = sum(loss)
        # 使用加速器 accelerator 对总的损失 total_loss 进行反向传播
        self.accelerator.backward(total_loss)
        # 返回每个损失 loss 的值
        return loss 

    # 训练一个 epoch 的模型，其中的参数 desc 表示进度条的描述信息  
    def train_one_epoch(self, desc):
        # 使用 tqdm 库初始化的进度条对象
        tbar = tqdm(
            self.train_loader, 
            desc=desc, 
            ncols=self.line_width, 
            disable= not self.accelerator.is_main_process
        )
        # 记录每个 batch 的损失
        epoch_loss = [] 
        for *feature, target in tbar: 
            # 使用ResNet model
            # feature, target = feature.cuda().half(), target.cuda()
            # 将模型参数梯度累加 self.config.train.gradient_accumulate_steps 次后再更新模型
            with self.accelerator.accumulate(self.model):
                # 获取模型的预测结果
                # coatnet
                prediction = self.forward_model(feature)
                # resnet
                #prediction = self.forward_model(feature)[:, :7]
                # inception_v3
                #prediction = self.forward_model(feature).logits[:, :7]
                #print(self.forward_model(feature).logits.shape)
                # 计算损失
                batch_loss = self.backward_model(prediction, target)
                # 更新模型参数
                self.model_optim.step()
                 # 更新学习率
                self.model_scheduler.step() 
                # 清除参数梯度
                self.model_optim.zero_grad() 
                # 将损失添加到 epoch_loss 中
                epoch_loss.append(batch_loss)
        # 将 epoch_loss 转换为 tensor，并返回
        return torch.tensor(epoch_loss)

    # 装饰器，表示在下面的代码块中不进行梯度计算，以减少内存消耗和提高运行速度
    @torch.no_grad() 
    def val_one_epoch(self):
        # 初始化一个空列表，用于存储每个 batch 的验证损失
        epoch_loss = [] 
        # 遍历验证集中的每个 batch
        for *feature, target in self.val_loader: 
            # 进行前向传播，生成预测值 pred
            # coatnet
            pred = self.forward_model(feature)
            #print(self.forward_model(feature))
            #print(self.forward_model(feature).logits.shape)
            #print(self.forward_model(feature).shape)
            # resnet  
            # inception_v3
            #pred = self.forward_model(feature)[:, :7]
            # 将当前 batch 的验证损失添加到 epoch_loss 列表
            epoch_loss.append(self.val_loss(pred, target))
        # auc = torch.tensor([auc])
        # 将 epoch_loss 列表转换为 tensor 并返回
        return torch.tensor(epoch_loss)
    
    # 保存模型的checkpoint，包含了当前模型的状态和优化器的状态以及调度器的状态
    def save_checkpoint(self, path):
        model = self.accelerator.unwrap_model(self.model)
        out = {'model': model.state_dict(), 'optim': self.model_optim.state_dict(), 'scheduler': self.model_scheduler}
        torch.save(out, path)
    
    # 加载预训练模型的函数
    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(checkpoint['model'])
        self.model_optim.load_state_dict(checkpoint['optim'])
        self.model_scheduler = checkpoint['scheduler']
    
    # 将训练好的模型保存到指定路径的功能
    def save_inference(self, path):
        model = self.accelerator.unwrap_model(self.model)
        out = {'model': model.state_dict()}
        torch.save(out, path)
    
    # 加载保存好的模型权重的
    def load_inference(self, path):
        out = torch.load(path)
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(out['model'])
        self.model = self.accelerator.prepare_model(model)

    # 打印分类器的参数总数的
    def describe_trainer(self):
        print(f'classifier with {sum(param.numel() for param in self.model.parameters()):,} params') 