import argparse
import os
import torch
from models.yolo import Model, Detect
from utils.torch_utils import prune
from utils.general import intersect_dicts
import yaml


def prune_and_finetune(args):
    # 1. 设定参数
    weights = args.weights
    cfg = args.cfg
    data = args.data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading model from {weights}...")

    # 2. 加载模型 (处理 weights_only 问题，如果 torch 版本较高)
    try:
        ckpt = torch.load(weights, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(weights, map_location=device)

    model = Model(cfg, ch=3, nc=args.nc).to(device)
    state_dict = ckpt['model'].float().state_dict()
    model.load_state_dict(state_dict, strict=False)

    # 3. 执行原生剪枝 (Sparsity 参数化)
    print(f"Executing utils.torch_utils.prune(model, {args.sparsity})...")
    prune(model, args.sparsity)

    # 4. 重新保存剪枝后的模型
    # 注意：这里的剪枝是"非结构化"的(权重置0)，必须配合微调才能恢复精度
    # 生成剪枝模型的保存路径，继承 weights 的同级路径
    weights_dir = os.path.dirname(weights)
    pruned_filename = "pruned_model.pt"
    pruned_path = os.path.join(weights_dir, pruned_filename)
    torch.save({'model': model}, pruned_path)
    print(f"Pruned model saved to {pruned_path}")

    # 5. 微调建议
    print("\n[IMPORTANT] You must now finetune this model.")
    print(
        f"Run: python train.py --img {args.img_size} --epochs {args.epochs} --weights {pruned_path} --cfg {cfg} --data {data} --name {args.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prune and finetune a YOLO model.")
    parser.add_argument('--weights', type=str, default='runs/train/yolov5b-anc-exp-test/weights/best.pt',
                        help='Path to the model weights')
    parser.add_argument('--cfg', type=str, default='models/yolov5b-anc-ghost.yaml',
                        help='Path to the model configuration file')
    parser.add_argument('--data', type=str, default='datasets/TWO/data.yaml',
                        help='Path to the data configuration file')
    parser.add_argument('--nc', type=int, default=4, help='Number of classes')
    parser.add_argument('--sparsity', type=float, default=0.3, help='Sparsity level for pruning')
    parser.add_argument('--img_size', type=int, default=64, help='Image size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for finetuning')
    parser.add_argument('--name', type=str, default='yolov5b-expx', help='Name for the finetuning run')

    args = parser.parse_args()
    prune_and_finetune(args)

# USEAGE: 
# python pruning.py --weights runs/train/yolov5b-anc-exp-test/weights/best.pt --cfg models/yolov5b-anc-ghost.yaml --data datasets/TWO/data.yaml --nc 4 --sparsity 0.3 --img_size 64 --epochs 50 --name yolov5b-expx