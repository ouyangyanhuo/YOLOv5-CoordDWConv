# YOLOv5-CoordDWConv

这是一个基于 Ultralytics YOLOv5 改造的轻量化目标检测项目。仓库在原版 YOLOv5 的训练、验证、推理、导出流程基础上，引入了 深度可分离坐标卷积核(CoordDWConv)，并额外提供了一个模型剪枝脚本 `pruning.py`，用于探索更轻量、更适合部署的检测网络。

从当前仓库内容来看，这个项目的核心方向是：

- 用深度可分离卷积替代部分标准下采样层，降低计算开销。
- 结合坐标注意力增强空间定位能力。
- 使用 5x5 大核和残差连接，兼顾感受野与稳定性。
- 保留 YOLOv5 常见工作流，便于继续训练、验证、推理和导出。

## 主要改动

### 1. 深度可分离坐标卷积核

仓库新增了两个核心模块：

- `models/CoordDWConv.py`
- `models/CoordDWConvPro.py`

它们都基于 Depthwise + Pointwise 结构，并融合了坐标注意力思想。

其中 `CoordDWConvPro` 的特点是：

- 使用 5x5 Depthwise Conv 扩大感受野。
- 使用横向/纵向池化构造坐标注意力。
- 当输入输出通道一致且步长为 1 时启用残差连接。

### 2. 剪枝脚本

- `pruning.py`

该脚本的作用是：

- 加载已有权重
- 对模型执行 `utils.torch_utils.prune(model, sparsity)` 非结构化剪枝
- 保存剪枝后的模型
- 给出后续微调命令

需要注意，这里做的是非结构化剪枝，剪枝后通常必须继续微调，否则精度很难稳定恢复。

## 训练流程

**注意：训练建议关闭早停机制，耐心训练，通常训练一个模型的完整周期在 24H 左右**

- 第一步：冷启动。

示例命令：
```bash
python train.py --cfg models/yolov5-CoordDWConv.yaml --data data/your_dataset.yaml --imgsz 64 --batch-size 512 --epochs 5000 --device 0 --name coorddwconv-exp
```

- 第二步：剪枝。

示例命令：
```bash
python pruning.py --weights runs/train/coorddwconv-exp/weights/best.pt --cfg models/yolov5-CoordDWConv.yaml --data data/your_dataset.yaml --nc 15 --sparsity 0.3 --img_size 64 --epochs 50 --name coorddwconv-pruned
```

- 第三步：微调。

示例命令：
```bash
python train.py --weights runs/train/coorddwconv-pruned/weights/best.pt --cfg models/yolov5-CoordDWConv.yaml --data data/your_dataset.yaml --imgsz 64 --batch-size 512 --epochs 5000 --device 0 --name coorddwconv-exp
```

- 第四步：模型导出为 ONNX → 使用 PNNX 转换为 NCNN

## 训练

基础训练命令示例：

```bash
python train.py --cfg models/yolov5-CoordDWConv.yaml --data data/your_dataset.yaml --imgsz 64 --batch-size 512 --epochs 5000 --device 0 --name coorddwconv-exp
```

说明：

- `--weights` 可替换为 `yolov5s.pt` 或你自己的预训练权重。
- 该仓库已有实验记录中使用过 `imgsz=64` 的小尺寸训练。
- 如果你的场景更接近常规目标检测任务，可以自行尝试更大的输入尺寸。

## 导出

导出 ONNX：

```bash
python export.py --weights runs/train/coorddwconv-exp/weights/best.pt --imgsz 64 64 --include onnx
```

仓库中的 `Example/` 目录已经包含示例导出结果：

- `Example/best.onnx`
- `Example/best.ncnn.param`
- `Example/best.ncnn.bin`

这说明项目本身有面向 ONNX / NCNN 部署的使用痕迹。

## 剪枝与微调

示例命令：

```bash
python pruning.py --weights runs/train/coorddwconv-exp/weights/best.pt --cfg models/yolov5-CoordDWConv.yaml --data data/your_dataset.yaml --nc 15 --sparsity 0.3 --img_size 64 --epochs 50 --name coorddwconv-pruned
```

执行后脚本会：

- 加载指定权重
- 执行非结构化剪枝
- 在权重目录下生成 `pruned_model.pt`
- 输出后续微调命令

推荐流程：

1. 先完成基线训练，得到 `best.pt`
2. 对 `best.pt` 执行剪枝
3. 用 `pruned_model.pt` 继续微调若干 epoch
4. 再次评估精度、速度和模型大小

## 已知问题与注意事项

### 当前环境需自行安装 PyTorch

如果本地环境未安装 `torch`，直接运行训练或模型导入会失败。开始前请先完成依赖安装。

## 适用场景

这个仓库更适合以下用途：

- 轻量化目标检测实验
- 深度可分离坐标卷积核效果验证
- 小模型导出与边缘部署探索
- YOLOv5 改造版结构研究

## 致谢

本项目建立在 Ultralytics YOLOv5 的基础之上，并在其训练、验证、推理、导出框架上进行了自定义改造。
