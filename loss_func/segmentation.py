import torch 
# PyTorch提供的一个指标库，用于评估训练过程和模型的性能
import torchmetrics 

# 定义两个loss函数：交叉熵损失和Dice损失
class CE_DICE(torch.nn.Module): 
    def __init__(self, num_classes, device): 
        super().__init__()
        # 定义交叉熵损失函数
        self.ce = torch.nn.CrossEntropyLoss()
        # 定义Dice系数指标，类别数和平均模式设置为num_classes和macro
        self.dice = torchmetrics.Dice(num_classes=num_classes, average='macro').to(device)
        # crossattention
        #self.dice = torchmetrics.Accuracy(num_classes=num_classes, average='macro').to(device)
        self.names = ['CE', 'DICE']

    # 对输入的预测值和目标值计算了两个损失函数的值，并以列表的形式返回这两个损失函数的值
    def forward(self, pred, target): 
        # 计算交叉熵损失
        ce = self.ce(pred, target) 
        # 计算Dice系数
        dice = self.dice(pred, target)
        # crossattention
        #acc = self.dice(pred,target)
        # 返回交叉熵损失和负的Dice系数，为了使用梯度下降优化这两个指标，让它们都最小化
        # crossattention
        return [ce, -dice] 

# 用于验证损失,计算交叉熵损失、AUC、Dice 分数和准确率指标
class ValLoss(torch.nn.Module): 
    def __init__(self, num_classes, device): 
        super().__init__()
        # 表示使用的设备
        self.device = device 
        # 计算AUC task='multiclass' 表示使用多类别分类任务，
        # num_classes=num_classes 表示分类的类别数，average='macro' 表示使用macro平均
        self.auc = torchmetrics.AUROC(task='multiclass', num_classes=num_classes, average='macro').to(self.device)
        # 计算CE
        self.ce = torch.nn.CrossEntropyLoss()
        # 计算DICE
        self.dice = torchmetrics.Dice(num_classes=num_classes, average='macro').to(device)
        # 计算ACC
        self.acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, average='macro').to(device)
        # 列表类型的成员变量，存储了每种指标的名称
        self.names = ['CE', 'AUC', 'DICE', 'ACC']

    # 输入为模型的输出 pred 和目标标签 target，输出为一个列表，包含了四种指标的值
    def forward(self, pred, target): 
        ce = self.ce(pred, target) 
        dice = self.dice(pred, target)
        acc = self.acc(pred, target)
        auc = self.auc(pred, target)
        return [ce, auc, dice, acc]
