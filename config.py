import json
from ml_collections import ConfigDict

# DefaultConfig类，用于存储训练过程中的各种配置信息
class DefaultConfig:

    # 定义一个列表，包含所有可配置字段
    FIELDS = [
        # 存储与CUDA相关的设置，是否启用cudnn，是否使用cudnn的benchmark模式，是否使用确定性算法等
        "cuda",
        # 用于存储与数据相关的设置，类别数量
        "data",
        # 用于存储各种路径相关的设置，数据文件夹路径，训练日志文件夹路径等
        "path",
        # 用于存储训练相关的设置，学习率，优化器，批次大小等
        "train",
        # 用于存储验证相关的设置，验证批次大小等
        "validation",
        # 用于存储报告相关的设置，报告的行宽等
        "report",
    ]

    def __init__(self):
        # 初始化一个ConfigDict实例
        self.cuda = ConfigDict()
        # 启用cudnn
        self.cuda.cudnn = True  # torch.backends.cudnn.enable
        # 不使用cudnn的benchmark模式
        self.cuda.benchmark = False  # torch.backends.cudnn.benchmark
        # 不使用确定性算法
        self.cuda.deterministic = False  # torch.backends.cudnn.deterministic
        # 不设置随机种子
        self.cuda.seed = None

        # 初始化一个ConfigDict实例
        self.data = ConfigDict()
        # 有7个类别
        self.data.num_classes = 7

        # 初始化一个ConfigDict实例
        self.path = ConfigDict()
        # Ubuntu22.04
        # 设置为record.json的路径
        # self.path.json_path = '/home/yolo/Study/network-model/network/network_training/record.json'
        # # 设置为已处理数据文件夹的路径
        # self.path.processed_data_folder = "/home/yolo/Study/network-model/organized"
        # # 设置为训练日志文件夹的路径
        # self.path.train_log_folder = "/home/yolo/Study/network-model/train_logs"

        # WSL 2.0
        self.path.json_path = "/home/yolo/Study/network-model/new_network_training/record.json"
        self.path.processed_data_folder = "/home/yolo/Study/network-model/organized"
        self.path.train_log_folder = "/home/yolo/Study/network-model/train_logs"

        # 初始化一个ConfigDict实例
        self.train = ConfigDict()
        # 最大学习率
        self.train.max_lr = 1e-4
        # 表示使用Adam优化器
        self.train.optimizer = "adam"
        #self.train.lr_scheduler = "onecycle"
        # 使用余弦退火学习率调度器
        self.train.lr_scheduler = "cosineannealing"
        # 3 TTTT 128   CCCC   256   CCTT 256
        # 4 TTTT 128   CCCC   256   CCTT 256
        # resnet 256
        # InceptionV3_3 256
        # Nodownsample
        # CCTT 16
        # 训练批次大小
        self.train.train_batch_size = 256
        # 更新批次大小
        self.train.update_batch_size = 256
        # 梯度累积的步数
        self.train.gradient_accumulate_steps = self.train.update_batch_size // self.train.train_batch_size
        # 训练批次总数
        self.train.batches = self.train.update_batch_size * 100
        # 表示总训练周期数100
        self.train.total_epochs = 100
        # 改变不同的模型
        self.train.case_name = "coatnet_CCTT_4"
        #self.train.case_name = "coatnet_CCTT_4"
        #self.train.case_name = "coatnet_CCTT_Nodownsample_3"
        #self.train.case_name = "coatnet_CCCC_Nodownsample_4"
        #self.train.case_name = "coatnet_TTTT_Nodownsample_3"
        #self.train.case_name = "Resnet_3"
        #self.train.case_name = "ResnetNopretrained_3"
        #self.train.case_name = "InceptionV3_3"
        #self.train.case_name = "InceptionV3Nopretrained_3"
        #self.train.case_name = "coatnet_CCTT_Nodownsample_4_dataone_1.2"
        #self.train.case_name = "coatnet_CCTT_4_crossattention"
        #self.train.case_name = "coatnet_CCCC_source_4"
        # 模型保存模式
        self.train.save_mode = "all"
        # 表示不恢复模型
        self.train.restore = 'none'

        # 初始化一个ConfigDict实例
        self.validation = ConfigDict()
        # 表示验证批次大小
        self.validation.val_batch_size = 32

        # 初始化一个ConfigDict实例
        self.report = ConfigDict()
        # 表示报告的行宽
        self.report.line_width = 100
    # 从 JSON 文件加载配置信息
    def load_json(self, filepath):
        config = json.load(open(filepath, "r"))
        # 遍历 DefaultConfig.FIELDS 中的每个字段，将读取到的配置值转换为 ConfigDict 类型，并使用 setattr 函数将其设置为当前实例的属性
        [setattr(self, each_field, ConfigDict(config[each_field])) for each_field in DefaultConfig.FIELDS]
        return self

    # 和将配置信息保存到 JSON 文件
    def save_json(self, filepath):
        # 遍历 DefaultConfig.FIELDS 中的每个字段，使用 getattr 函数获取当前实例的属性值，然后调用 to_dict 方法将其转换为字典类型，最后将结果存储在 config 字典中
        config = {each_field: getattr(self, each_field).to_dict() for each_field in DefaultConfig.FIELDS}

        with open(filepath, "w") as fileout:
            # 将 config 字典保存为 JSON 文件，ensure_ascii=False 表示允许非 ASCII 字符，indent=4 表示使用四个空格进行缩进
            json.dump(config, fileout, ensure_ascii=False, indent=4)
        return self
