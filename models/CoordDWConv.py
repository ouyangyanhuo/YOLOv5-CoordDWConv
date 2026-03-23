import torch
import torch.nn as nn

class Hardsigmoid(nn.Module):
    # NCNN 友好的 H-Sigmoid，替代标准 Sigmoid
    def __init__(self, inplace=True):
        super(Hardsigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        # 模拟 Sigmoid: (x + 3) / 6 并在 0-1 截断
        return self.relu(x + 3) / 6

class CoordDWConv(nn.Module):
    # CoordDWConv
    # 1. 5x5 大核 + 残差 (保持精度)
    # 2. H-Sigmoid + ReLU (NCNN 提速)
    # 3. 去除冗余 Bias (更干净)
    def __init__(self, c1, c2, k=5, s=1, p=None, g=1, act=True, reduction=16):
        super(CoordDWConv, self).__init__()
        
        # === 主干路径 (保持高精度配置) ===
        self.dw_conv = nn.Conv2d(c1, c1, kernel_size=k, stride=s, padding=k//2 if p is None else p, groups=c1, bias=False)
        self.pw_conv = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        # 主干保留 Hardswish，因为这里包含主要特征信息
        self.act = nn.Hardswish() if act else nn.Identity()

        # === 坐标注意力路径 (极致提速) ===
        mip = max(8, c2 // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        # 优化点1: bias=False (因为后面紧跟 BN，Bias 是多余的)
        self.conv_pool = nn.Conv2d(c2, mip, kernel_size=1, stride=1, padding=0, bias=False)
        self.gn = nn.BatchNorm2d(mip) 
        
        # 优化点2: 内部使用 ReLU 代替 Hardswish
        # 注意力掩码的生成不需要太复杂的非线性，ReLU 更快
        self.act_att = nn.ReLU() 
        
        self.conv_h = nn.Conv2d(mip, c2, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, c2, kernel_size=1, stride=1, padding=0)
        
        # 优化点3: 使用 Hardsigmoid 替代 Sigmoid
        # 消除 exp() 运算，对低频 CPU 极其友好
        self.sigmoid = Hardsigmoid()

        self.use_res = (c1 == c2) and (s == 1)

    def forward(self, x):
        # 主干计算
        feat = self.dw_conv(x)
        feat = self.pw_conv(feat)
        feat = self.bn(feat)
        feat = self.act(feat)
        
        # 坐标注意力计算
        identity_feat = feat
        n, c, h, w = feat.size()
        
        x_h = self.pool_h(feat)
        x_w = self.pool_w(feat).permute(0, 1, 3, 2)
        
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv_pool(y)
        y = self.gn(y)
        y = self.act_att(y) # 使用 ReLU
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        # 使用 H-Sigmoid
        a_h = self.sigmoid(self.conv_h(x_h))
        a_w = self.sigmoid(self.conv_w(x_w))
        
        out = identity_feat * a_w * a_h

        if self.use_res:
            return x + out
        else:
            return out