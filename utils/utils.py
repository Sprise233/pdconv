import json
from typing import Union

import yaml
import torch

import torch.nn as nn
if torch.__version__.startswith('2.'):
    # 使用 PyTorch 2.x 的实现
    from torch._dynamo import OptimizedModule


def load_json(json_path):
    with open(json_path, 'r') as file:
        json_dict = json.load(file)

    return json_dict

def load_yaml(yaml_path):
    # 读取 YAML 文件并转换为字典
    with open(yaml_path, 'r',  encoding='utf-8') as file:
        yaml_dict = yaml.safe_load(file)

    return yaml_dict


def get_model_params_count(model: nn.Module):
    """
    获取模型的总参数量、可训练参数量和不可训练参数量

    :param model: 传入的PyTorch模型
    :return: 返回一个字典，包含 total_params, trainable_params, non_trainable_params
    """
    # 计算总参数量
    total_params = sum(p.numel() for p in model.parameters())

    # 计算可训练参数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 计算不可训练参数量
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params
    }


import torch.nn.functional as F

def resize_image_tensor(image_tensor, size, mode='bilinear'):
    """
    缩放五维 Tensor 的高度 (H) 和宽度 (W)，保持 batch (B)、通道数 (C) 和深度 (L) 不变。

    参数:
    - image_tensor (Tensor): 形状为 (B, C, L, H, W) 的图像 Tensor
    - size (tuple): 目标尺寸 (height, width)，例如 (新高度, 新宽度)
    - mode (str): 插值模式，默认是 'bilinear'，可选 'nearest', 'bilinear', 'bicubic', 'trilinear' 等。

    返回:
    - 缩放后的图像 Tensor
    """
    # 确保输入是五维 Tensor: (B, C, L, H, W)
    assert image_tensor.ndim == 5, "输入的图像 Tensor 应该是五维的 (B, C, L, H, W)"

    # 将 (B, C, L, H, W) reshape 为 (B * C * L, H, W) 以只缩放 H 和 W
    b, c, l, h, w = image_tensor.shape
    image_tensor_reshaped = image_tensor.view(b * c * l, h, w)

    # 对 H 和 W 进行缩放
    resized_tensor = F.interpolate(image_tensor_reshaped.unsqueeze(1), size=size, mode=mode, align_corners=False)

    # 将 Tensor reshape 回 (B, C, L, new_H, new_W)
    resized_tensor = resized_tensor.squeeze(1).view(b, c, l, size[0], size[1])

    return resized_tensor

import torch

def load_model_weights(network, filename_or_checkpoint, device, is_ddp=False):
    """
    加载模型的权重并应用到网络模型中。

    参数:
    - network: 要加载权重的网络模型
    - filename_or_checkpoint: 文件名或者检查点字典
    - device: 设备（例如 'cpu' 或 'cuda'）
    - is_ddp: 是否使用了分布式数据并行 (DDP)
    """

    # 加载检查点
    if isinstance(filename_or_checkpoint, str):
        checkpoint = torch.load(filename_or_checkpoint, map_location=device)
    else:
        checkpoint = filename_or_checkpoint

    # 处理 DataParallel 模型
    new_state_dict = {}
    for k, value in checkpoint['network_weights'].items():
        key = k
        if key not in network.state_dict().keys() and key.startswith('module.'):
            key = key[7:]
        new_state_dict[key] = value

    # 将权重加载到模型中
    if torch.__version__.startswith('1.'):
        if is_ddp:
            # 直接加载状态字典，无需检查 OptimizedModule
            network.module.load_state_dict(new_state_dict)
        else:
            # 直接加载状态字典，无需检查 OptimizedModule
            network.load_state_dict(new_state_dict)
    else:
        if is_ddp:
            if isinstance(network.module, OptimizedModule):
                network.module._orig_mod.load_state_dict(new_state_dict)
            else:
                network.module.load_state_dict(new_state_dict)
        else:
            if isinstance(network, OptimizedModule):
                network._orig_mod.load_state_dict(new_state_dict)
            else:
                network.load_state_dict(new_state_dict)

    return network
