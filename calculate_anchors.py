# calculate_anchors.py
from utils.autoanchor import kmean_anchors

# 配置参数
my_data_yaml = 'datasets/zhuxian/data.yaml'     # 这里换成你自己的yaml文件路径
img_size = 64                                   # 你的训练图片尺寸
num_anchors = 3                                 # 默认是9个anchors (3层feature map * 3个)

# 执行计算
print(f"正在为 {my_data_yaml} 计算最佳 Anchors...")
kmean_anchors(
    dataset=my_data_yaml, 
    n=num_anchors, 
    img_size=img_size, 
    thr=4.0, 
    gen=1000, 
    verbose=True
)