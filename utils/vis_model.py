import sys

import netron
import onnx
import onnxoptimizer
import torch
import torchprofile
import torch.nn as nn
from matplotlib import pyplot as plt
from onnxruntime.transformers.torch_onnx_export_helper import TrainingMode
from fvcore.nn import FlopCountAnalysis

def vis_model_from_params(save_folder, dummy_input_shape, architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import, num_input_channels, num_output_channels, enable_deep_supervision):
    # 生成 view_model.py 文件的代码
    code = f"""
import sys

import onnx
import torch

import netron
import onnxoptimizer
from fvcore.nn import FlopCountAnalysis

from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from onnxruntime.transformers.torch_onnx_export_helper import TrainingMode

# 定义要可视化的模型（例如 ResNet18）
model = get_network_from_plans(arch_class_name='{architecture_class_name}',
                               arch_kwargs={arch_init_kwargs},
                               arch_kwargs_req_import={arch_init_kwargs_req_import},
                               input_channels={num_input_channels},
                               output_channels={num_output_channels},
                               allow_init=True,
                               deep_supervision={enable_deep_supervision})
                               
# 定义输入张量的尺寸（例如输入是3通道的224x224图像）
dummy_input = torch.randn({dummy_input_shape})

# 计算总参数量
total_params = sum(p.numel() for p in model.parameters())
print('total params is : ', total_params)

# 使用 fvcore 计算 FLOPs
flop_analysis = FlopCountAnalysis(model, dummy_input)

# 计算模型的总 FLOPs
total_flops = flop_analysis.total()

# 转换为 GFLOPs (1 GFLOPs = 1e9 FLOPs)
total_gflops = total_flops / 1e9

print("Total GFLOPs: " + str(round(total_gflops, 4)) + " GFLOPs")

# 将模型导出为 ONNX 文件
onnx_file_path = './model.onnx'
torch.onnx.export(model, dummy_input, onnx_file_path, training=TrainingMode.TRAINING)
onnx.save(onnx.shape_inference.infer_shapes(onnx.load(onnx_file_path)), onnx_file_path)

# 使用 Netron 可视化模型
netron.start(onnx_file_path)

# 加载模型
model = onnx.load(onnx_file_path)
# 使用 onnxoptimizer 进行优化，删除 Identity 层
passes = ["eliminate_identity"]  # 定义要应用的优化 passes
optimized_model = onnxoptimizer.optimize(model, passes)

# 保存优化后的模型
onnx.save(optimized_model, onnx_file_path)

# 等待用户按下回车键以结束进程
input("按下回车键结束...")
netron.stop()
print("已结束。")
sys.exit()
    """

    # 将代码写入文件
    with open(f'{save_folder}/view_model.py', 'w') as f:
        f.write(code)

    print(f"Python file '{save_folder}view_model.py' has been generated.")

def vis_model_from_class(dummy_input, model, save_folder: str=None):
    compute_flops_percentage_by_module(model, dummy_input)

    # 将模型导出为 ONNX 文件
    onnx_file_path =  './model.onnx' if save_folder is None else save_folder
    torch.onnx.export(model, dummy_input, onnx_file_path, training=TrainingMode.TRAINING)

    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(onnx_file_path)), onnx_file_path)

    # 使用 Netron 可视化模型
    netron.start(onnx_file_path)

    # 加载模型
    model = onnx.load(onnx_file_path)
    # 使用 onnxoptimizer 进行优化，删除 Identity 层
    passes = ["eliminate_identity"]  # 定义要应用的优化 passes
    optimized_model = onnxoptimizer.optimize(model, passes)

    # 保存优化后的模型
    onnx.save(optimized_model, onnx_file_path)

    # 等待用户按下回车键以结束进程
    input("按下回车键结束...")
    netron.stop()
    print("已结束。")
    sys.exit(0)


def flatten_modules(model: torch.nn.Module, prefix='', max_depth=2, current_depth=0):
    """
    获取模型层次结构，只递归两层，并返回模块的名称和相应模块
    """
    modules = []
    if current_depth >= max_depth:
        return modules
    for name, module in model.named_children():
        full_name = prefix + ('.' if prefix else '') + name
        modules.append((full_name, module))
        # 继续递归子模块
        modules.extend(flatten_modules(module, full_name, max_depth, current_depth + 1))
    return modules


def compute_flops_percentage_by_module(model: torch.nn.Module, input_tensor: torch.Tensor, max_depth=2,
                                       min_percentage=1):
    # 计算总参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"total params is : {total_params}")

    # 使用 fvcore 计算每个模块的 FLOPs
    flop_analysis = FlopCountAnalysis(model, input_tensor)

    # 计算模型的总 FLOPs 并转换为 GFLOPs
    total_flops = flop_analysis.total()
    total_gflops = total_flops / 1e9
    print(f"Total FLOPs: {total_gflops:.4f} GFLOPs")

    # 获取模型层次结构并限制深度
    def flatten_modules(model: torch.nn.Module, prefix='', max_depth=2, current_depth=0):
        modules = []
        if current_depth >= max_depth:
            return modules
        for name, module in model.named_children():
            full_name = prefix + ('.' if prefix else '') + name
            modules.append((full_name, module))
            # 继续递归子模块
            modules.extend(flatten_modules(module, full_name, max_depth, current_depth + 1))
        return modules

    modules = flatten_modules(model, max_depth=max_depth)

    # 获取各模块的 FLOPs 数据
    flops_dict = flop_analysis.by_module()
    selected_flops_dict = {name: flops_dict.get(name, 0) for name, _ in modules}

    # 提取模块名称和对应的 FLOPs
    module_names = list(selected_flops_dict.keys())
    flops = [v for v in selected_flops_dict.values()]

    # 计算每个模块的 FLOPs 占比
    flops_percentage = [(f / total_flops) * 100 for f in flops]

    # 处理占比过小的模块，将小于 min_percentage 的部分合并为“其他”
    major_modules = []
    major_flops = []
    other_flops = 0

    for name, percentage, flop in zip(module_names, flops_percentage, flops):
        if percentage >= min_percentage:
            major_modules.append(name)
            major_flops.append(flop)
        else:
            other_flops += flop

    if other_flops > 0:
        major_modules.append("Others")
        major_flops.append(other_flops)

    # 计算新的 FLOPs 占比
    major_flops_percentage = [(f / total_flops) * 100 for f in major_flops]

    # 可视化 FLOPs 占比（饼图）
    plt.figure(figsize=(10, 7))
    plt.pie(major_flops_percentage, labels=major_modules, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
    plt.title('FLOPs Proportion by Module (Up to 2 Layers)')
    plt.axis('equal')  # 确保饼图为圆形
    plt.show()

    # 如果有合并到“其他”的模块，单独用条形图展示
    if other_flops > 0:
        # 提取小模块并展示
        minor_modules = [name for name, percentage in zip(module_names, flops_percentage) if
                         percentage < min_percentage]
        minor_flops = [flop for name, percentage, flop in zip(module_names, flops_percentage, flops) if
                       percentage < min_percentage]
        minor_flops_percentage = [(f / total_flops) * 100 for f in minor_flops]

        plt.figure(figsize=(10, 7))
        plt.barh(minor_modules, minor_flops_percentage, color='skyblue')
        plt.xlabel('FLOPs Percentage')
        plt.title('Minor Modules (FLOPs Percentage Less Than 1%)')
        plt.tight_layout()
        plt.show()
# if __name__ == '__main__':
#     vis_model('')