import torch
# 导入PyTorch的nn模块，它提供了构建神经网络的类和函数
import torch.nn as nn

# 从einops库中导入rearrange函数，用于改变张量的形状和维度
from einops import rearrange
 # 从einops库的layers.torch模块中导入Rearrange类，用于在神经网络中改变张量的形状和维度
from einops.layers.torch import Rearrange


# https://github.com/chinhsuanwu/coatnet-pytorch/blob/master/coatnet.py

# 定义了一个包含3x3卷积、BatchNorm和GELU激活函数的序列模块
def conv_3x3_bn(inp, oup, image_size, downsample=False):
    # 如果downsample参数为False，步长设置为1，否则设置为2
    stride = 1 if downsample == False else 2
    return nn.Sequential(
        # 创建一个二维卷积层，输入通道数为inp，输出通道数为oup，卷积核尺寸为3，步长为stride，padding为1，不使用偏置项
        #print("inp",inp,"out",oup),
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        #print("inp",inp,"out",oup),
        # 创建一个二维BatchNorm层，用于对oup个通道进行归一化处理
        nn.BatchNorm2d(oup),
        # 创建一个GELU激活层
        nn.GELU()
    )

# 定义了一个预归一化（Pre-Normalization）模块,在前向传播时先对输入进行归一化，然后再进行前向传播函数
class PreNorm(nn.Module):
    # 初始化函数，接受三个参数：dim表示输入的维度，fn表示前向传播函数，norm表示归一化函数
    def __init__(self, dim, fn, norm):
        super().__init__()
        # 创建一个对dim维输入进行归一化的层
        self.norm = norm(dim)
        # 前向传播函数
        self.fn = fn
    # 前向传播函数
    def forward(self, x, **kwargs):
        # 先对输入x进行归一化处理，然后传入前向传播函数fn
        return self.fn(self.norm(x), **kwargs)

# SE模块是Squeeze-and-Excitation，它用于重新校准特征图的通道权重，增强网络的表示能力。
class SE(nn.Module):
    def __init__(self, inp, oup, expansion=0.25):
        super().__init__()
        # 创建一个自适应平均池化层，输出尺寸为1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 创建一个全连接网络
        self.fc = nn.Sequential(
            # 线性层，输出维度为inp*expansion，不使用偏置项
            nn.Linear(oup, int(inp * expansion), bias=False),
            # GELU激活层
            nn.GELU(),
            # 线性层，输出维度为oup，不使用偏置项
            nn.Linear(int(inp * expansion), oup, bias=False),
            # Sigmoid激活层
            nn.Sigmoid()
        )

    def forward(self, x):
        # 获得输入的尺寸
        b, c, _, _ = x.size()
        # 对输入x进行平均池化，然后改变形状
        y = self.avg_pool(x).view(b, c)
         # 对y进行全连接网络处理，然后改变形状
        y = self.fc(y).view(b, c, 1, 1)
        # 返回x和y的点积
        return x * y

#FeedForward模块是前馈神经网络模块的简称，它由两个全连接层和一个GELU激活函数组成
#中间还插入了一个Dropout层来防止过拟合。
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        # 创建一个序列化的神经网络
        self.net = nn.Sequential(
            # 线性层，输出维度为hidden_dim
            nn.Linear(dim, hidden_dim),
            # GELU激活层
            nn.GELU(),
            # Dropout层，丢弃率为dropout
            nn.Dropout(dropout),
            # 线性层，输出维度为dim
            nn.Linear(hidden_dim, dim),
            # Dropout层，丢弃率为dropout
            nn.Dropout(dropout)
        )

    def forward(self, x):
         # 对输入x进行前向传播
        return self.net(x)


class MBConv(nn.Module):
    # 初始化函数，inp是输入通道数，oup是输出通道数，image_size是输入图像的大小
    # downsample决定是否下采样，expansion是扩展因子，用于控制卷积层中间的通道数
    def __init__(self, inp, oup, image_size, downsample=False, expansion=4):
        super().__init__()
        # 保存下采样标志
        self.downsample = downsample
        # 决定步长
        stride = 1 if self.downsample == False else 2
        # 计算隐藏层的维度（即中间通道数）
        hidden_dim = int(inp * expansion)

        # 如果进行下采样，那么就定义最大池化层（用于下采样）和1x1的卷积层（用于改变通道数）
        if self.downsample:
            self.pool = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        # 如果扩展因子为1，那么定义一个深度卷积（DW）和一个点卷积（PW）
        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                # 深度卷积用于提取特征
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # pw-linear
                # 点卷积用于改变通道数
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        # 如果扩展因子不为1，那么定义一个点卷积，一个深度卷积和一个SE模块
        # 点卷积用于改变通道数，深度卷积用于提取特征，SE模块用于进行通道注意力
        else:
            self.conv = nn.Sequential(
                # pw
                # down-sample in the first conv
                nn.Conv2d(inp, hidden_dim, 1, stride, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1,
                          groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                SE(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        # 定义的卷积序列包装进一个PreNorm模块，该模块会先进行批标准化，然后将结果输入给卷积序列
        self.conv = PreNorm(inp, self.conv, nn.BatchNorm2d)

    # 在前向传播函数中，如果需要下采样，就先进行最大池化，然后通过1x1的卷积改变通道数，与卷积序列的输出相加；
    # 如果不需要下采样，就直接将输入和卷积序列的输出相加。这实现了残差连接，可以帮助模型更好地进行反向传播，
    # 提高模型的性能。
    def forward(self, x):
        if self.downsample:
            return self.proj(self.pool(x)) + self.conv(x)
        else:
            return x + self.conv(x)

# 实现了注意力机制的模块。注意力机制是一种用于提升模型性能的技术，
# 它使模型能够根据输入数据的不同部分分配不同的注意力
class Attention(nn.Module):
    # 首先指定了输入和输出的维度，图片的大小，以及注意力头的数量和每个头的维度。
    # 注意力头的数量和每个头的维度用于计算注意力机制的内部维度。
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, dropout=0.):
        super().__init__()
        # 计算了注意力模块内部的维度
        inner_dim = dim_head * heads
        # 决定是否需要进行输出投影
        # 如果注意力头的数量为1且每个头的维度与输入维度相同，那么输出的维度就会与输入维度相同，
        # 此时不需要进行输出投影。在其他情况下，为了保证输出的维度与输入一致，需要通过一个线性变换进行输出投影。
        project_out = not (heads == 1 and dim_head == inp)

        # 图片的高度和宽度
        self.ih, self.iw = image_size

        # 注意力头的数量
        self.heads = heads
        # 缩放因子，用于缩放注意力得分
        self.scale = dim_head ** -0.5

        # parameter table of relative position bias
        # 定义了相对位置偏置表，它是一个可训练的参数，用于计算相对位置偏置
        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * self.ih - 1) * (2 * self.iw - 1), heads))

        # 计算每个像素位置与其他所有像素位置的相对坐标
        coords = torch.meshgrid((torch.arange(self.ih), torch.arange(self.iw)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]

        relative_coords[0] += self.ih - 1
        relative_coords[1] += self.iw - 1
        relative_coords[0] *= 2 * self.iw - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        # self.attend是一个softmax函数，用于将注意力得分转化为概率分布；
        # self.to_qkv是一个线性层，用于将输入转化为查询（Q）、键（K）和值（V）
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, oup),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    # 前向传播函数中，首先计算查询、键和值，然后将其分别拆分为不同的注意力头
    # 计算查询和键的点积，然后乘以缩放因子得到原始的注意力得分。
    # 然后，通过查表得到相对位置偏置，并添加到原始的注意力得分上。
    def forward(self, x):
         # 将输入 x 传入到一个线性变换层 (self.to_qkv)，然后分别得到 query、key、value 三个张量
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # 将得到的 query、key、value 三个张量按照头数 (self.heads) 重新排列，以便进行多头注意力计算
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        # 计算 query 和 key 的点积，然后乘以缩放因子 (self.scale) 得到原始的注意力得分
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Use "gather" for more efficiency on GPUs
        # 从相对位置偏置表中收集相应的偏置，并将其加到原始的注意力得分上
        relative_bias = self.relative_bias_table.gather(
            0, self.relative_index.repeat(1, self.heads))
        relative_bias = rearrange(
            relative_bias, '(h w) c -> 1 c h w', h=self.ih*self.iw, w=self.ih*self.iw)
        dots = dots + relative_bias

        # 对得到的注意力得分进行 softmax 运算，得到最终的注意力权重
        attn = self.attend(dots)
        # 用得到的注意力权重去加权平均 value，得到最终的输出
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
         # 如果设置了输出线性变换层 (self.to_out)，则将输出传入该层得到最终结果，否则直接返回
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, downsample=False, dropout=0.):
        super().__init__()
        # 计算隐藏层的维度，通常为输入维度的 4 倍
        hidden_dim = int(inp * 4)

        # 存储输入图像的尺寸
        self.ih, self.iw = image_size
        # 是否进行下采样
        self.downsample = downsample

        # 如果进行下采样，定义两个最大池化层和一个卷积层作为投影层  
        if self.downsample:
            self.pool1 = nn.MaxPool2d(3, 2, 1)
            self.pool2 = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        # 定义自注意力层和前馈神经网络层
        self.attn = Attention(inp, oup, image_size, heads, dim_head, dropout)
        self.ff = FeedForward(oup, hidden_dim, dropout)
        
        # 对自注意力层和前馈神经网络层进行重排和正则化
        self.attn = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            PreNorm(inp, self.attn, nn.LayerNorm),
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )

        self.ff = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            PreNorm(oup, self.ff, nn.LayerNorm),
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )

    # 在前向传播函数中，根据是否进行下采样来决定如何处理输入
    def forward(self, x):
        # 如果进行下采样，先通过池化层和投影层，然后加上自注意力的结果
        if self.downsample:
            x = self.proj(self.pool1(x)) + self.attn(self.pool2(x))
        # 如果进行下不采样，加上前馈神经网络的结果
        else:
            x = x + self.attn(x)
        x = x + self.ff(x)
        return x

# CoAtNet,基于卷积和自注意力的网络结构
class CoAtNet(nn.Module):
    def __init__(self, image_size, in_channels, num_blocks, channels, num_classes=1000, block_types=['C', 'C', 'T', 'T']):
        super().__init__()
        # 图像的尺寸
        ih, iw = image_size
        # 定义两种类型的块：MBConv和Transformer
        block = {'C': MBConv, 'T': Transformer}
        # 定义一个可学习的参数，用于在最后的全连接层之后进行维度缩放

        self.temp = torch.nn.Parameter(torch.tensor(2.0), requires_grad=True)
        

        # 定义了网络的五个阶段，每个阶段由一系列块组成。每个阶段的输入和输出通道数，
        # 块的数量和类型，以及图像的尺寸都是可配置的。最后，定义了一个全连接层，用于将网络的输出转化为类别预测。
        self.s0 = self._make_layer(
            conv_3x3_bn, in_channels, channels[0], num_blocks[0], (ih // 2, iw // 2))
        self.s1 = self._make_layer(
            block[block_types[0]], channels[0], channels[1], num_blocks[1], (ih // 4, iw // 4))
        self.s2 = self._make_layer(
            block[block_types[1]], channels[1], channels[2], num_blocks[2], (ih // 8, iw // 8))
        self.s3 = self._make_layer(
            block[block_types[2]], channels[2], channels[3], num_blocks[3], (ih // 16, iw // 16))
        self.s4 = self._make_layer(
            block[block_types[3]], channels[3], channels[4], num_blocks[4], (ih // 32, iw // 32))

        self.fc = nn.Linear(channels[-1], num_classes, bias=False)

    def forward(self, x):
        # 通过网络的每个阶段
        x = self.s0(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)

        # 在空间维度上进行平均池化，然后通过全连接层
        x = x.mean(dim=(-2, -1))
        x = self.fc(x)
        # 进行缩放，然后返回输出
        return x / self.temp

    # 生成一组网络层的。它接收五个参数，分别为：块类型、输入通道数、输出通道数、深度和图像尺寸
    def _make_layer(self, block, inp, oup, depth, image_size):
        # 初始化一个空的 nn.ModuleList 对象
        layers = nn.ModuleList([])
        for i in range(depth):
            # 如果是第一个块，那么将输入通道数设置为 inp，输出通道数设置为 oup，并开启下采样。
            if i == 0:
                layers.append(block(inp, oup, image_size, downsample=True))
                #print(layers[0])
                #layers.append(block(inp, oup, image_size, downsample=False))
            # 如果不是第一个块，那么输入和输出通道数都设为 oup   
            else:
                layers.append(block(oup, oup, image_size))
                #print(layers[1])
        # 所有的块都生成完毕之后，使用 nn.Sequential 将 layers 列表中的所有块连接起来，
        # 形成一个新的网络层，然后返回这个网络层
        return nn.Sequential(*layers)

# 创建了一个基于 CoAtNet 的神经网络模型，并返回该模型
def coatnet_0():
    # 定义了模型的各个阶段 (stage) 中的块数量。这个模型总共有五个阶段，分别包含 2、2、3、5 和 2 个块
    num_blocks = [2, 2, 3, 5, 2]            # L
    # 定义了模型各个阶段的输出通道数。这个模型的第一个阶段输出 64 个通道，第二个阶段输出 96 个通道，
    channels = [64, 96, 192, 384, 768]      # D
    # 自定义的神经网络类，它接受五个参数：图像尺寸、输入通道数、各个阶段的块数量、各个阶段的输出通道数以及分类数
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=1000)


def coatnet_1():
    num_blocks = [2, 2, 6, 14, 2]           # L
    channels = [64, 96, 192, 384, 768]      # D
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=1000)


def coatnet_2():
    num_blocks = [2, 2, 6, 14, 2]           # L
    channels = [128, 128, 256, 512, 1026]   # D
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=1000)


def coatnet_3():
    num_blocks = [2, 2, 6, 14, 2]           # L
    channels = [192, 192, 384, 768, 1536]   # D
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=1000)


def coatnet_4():
    num_blocks = [2, 2, 12, 28, 2]          # L
    channels = [192, 192, 384, 768, 1536]   # D
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=1000)

# 接受一个PyTorch模型作为输入，并返回该模型中需要计算梯度的所有参数的数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # 定义一个随机输入张量img
    img = torch.randn(1, 3, 224, 224)

    net = coatnet_0()
    out = net(img)
    print(out.shape, count_parameters(net))

    net = coatnet_1()
    out = net(img)
    print(out.shape, count_parameters(net))

    net = coatnet_2()
    out = net(img)
    print(out.shape, count_parameters(net))

    net = coatnet_3()
    out = net(img)
    print(out.shape, count_parameters(net))

    net = coatnet_4()
    out = net(img)
    print(out.shape, count_parameters(net))