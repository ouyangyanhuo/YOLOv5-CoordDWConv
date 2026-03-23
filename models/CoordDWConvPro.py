import torch
import torch.nn as nn

class CoordDWConvPro(nn.Module):
    # CoordDWConv_Pro: 5x5 大核 + 坐标注意力 + 强制残差
    def __init__(self, c1, c2, k=5, s=1, p=None, g=1, act=True, reduction=16):
        # 注意：默认 kernel 改为 5，reduction 改为 16 (避免通道压缩过猛)
        super(CoordDWConvPro, self).__init__()
        
        # 1. 升级为 5x5 DWConv (增大感受野，这对小图定位至关重要)
        # NCNN 对 5x5 DW 有很好的优化，算力增加极少
        self.dw_conv = nn.Conv2d(c1, c1, kernel_size=k, stride=s, padding=k//2 if p is None else p, groups=c1, bias=False)
        self.pw_conv = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Hardswish() if act else nn.Identity()

        # 2. 坐标感知路径
        mip = max(8, c2 // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        self.conv_pool = nn.Conv2d(c2, mip, kernel_size=1, stride=1, padding=0)
        self.gn = nn.BatchNorm2d(mip) # GN 在小 Batch 下更稳
        
        self.conv_h = nn.Conv2d(mip, c2, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, c2, kernel_size=1, stride=1, padding=0)

        # 3. 残差标志位
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
        y = self.act(y)
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        
        out = identity_feat * a_w * a_h

        # 强制残差连接：如果注意力把特征算废了，原始特征还能救回来
        if self.use_res:
            return x + out
        else:
            return out