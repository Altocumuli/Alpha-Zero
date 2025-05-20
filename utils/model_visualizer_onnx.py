import torch
import onnx
import os
import sys
import numpy as np
import json
import webbrowser
from pathlib import Path
import netron

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.example_net import MyNet, BaseNetConfig
from env.go.go_env import GoGame

def visualize_mynet_with_onnx(open_browser=True):
    """
    使用ONNX格式可视化MyNet模型结构
    
    参数:
        open_browser: 是否自动在浏览器中打开可视化结果
    """
    # 创建保存目录
    viz_dir = Path("model_visualization")
    viz_dir.mkdir(exist_ok=True)
    
    # 创建与训练时相同的围棋环境
    env = GoGame(7)
    
    # 创建与训练时一致的模型配置
    model_config = BaseNetConfig(
        num_channels=256,
        dropout=0.3,
        linear_hidden=[256, 128]
    )
    
    # 检测设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建模型实例
    model = MyNet(env.observation_size, env.action_space_size, model_config, device=device)
    model.eval()  # 设置为评估模式
    
    # 尝试加载已有的模型参数（如果存在）
    try:
        checkpoint_path = "checkpoint/my_net_7x7_3layers_exfeat_1/best.pth.tar"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['state_dict'])
            print(f"成功加载模型参数: {checkpoint_path}")
        else:
            print(f"未找到模型参数文件: {checkpoint_path}")
            print("使用随机初始化的模型参数进行可视化")
    except Exception as e:
        print(f"加载模型参数时出错: {e}")
        print("使用随机初始化的模型参数进行可视化")
    
    # 创建一个与环境观察空间匹配的随机输入张量
    dummy_input = torch.randn(1, *env.observation_size).to(device)
    print(f"输入张量形状: {dummy_input.shape}")
    
    # 定义ONNX输出路径
    onnx_path = viz_dir / "mynet_model.onnx"
    
    # 导出ONNX模型
    torch.onnx.export(
        model,                  # 要导出的模型
        dummy_input,            # 模型输入
        onnx_path,              # 保存路径
        export_params=True,     # 保存模型参数
        opset_version=12,       # ONNX版本
        do_constant_folding=True,  # 常量折叠优化
        input_names=['input'],  # 输入名称
        output_names=['policy', 'value'],  # 输出名称
        dynamic_axes={
            'input': {0: 'batch_size'},
            'policy': {0: 'batch_size'},
            'value': {0: 'batch_size'}
        }
    )
    
    # 验证ONNX模型
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f"ONNX模型验证成功")
    except Exception as e:
        print(f"ONNX模型验证失败: {e}")
    
    print(f"ONNX模型已成功导出到: {onnx_path}")
    
    # 创建一个详细的模型分析
    model_info = {
        "model_name": "MyNet",
        "board_size": "7x7",
        "nodes_count": len(onnx_model.graph.node),
        "inputs": [{"name": i.name, "shape": [d.dim_value for d in i.type.tensor_type.shape.dim]} 
                  for i in onnx_model.graph.input],
        "outputs": [{"name": o.name, "shape": [d.dim_value for d in o.type.tensor_type.shape.dim]}
                   for o in onnx_model.graph.output],
        "operations": {}
    }
    
    # 统计操作类型
    for node in onnx_model.graph.node:
        op_type = node.op_type
        if op_type in model_info["operations"]:
            model_info["operations"][op_type] += 1
        else:
            model_info["operations"][op_type] = 1
    
    # 将模型信息保存为JSON
    with open(viz_dir / "model_analysis.json", "w") as f:
        json.dump(model_info, f, indent=4)
    
    # 创建一个简单的HTML可视化页面
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>MyNet 模型分析</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ max-width: 1000px; margin: 0 auto; }}
            .card {{ background: #f8f9fa; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            h1, h2 {{ color: #333; }}
            .chart {{ height: 300px; margin-top: 20px; }}
            pre {{ background: #eee; padding: 10px; border-radius: 4px; overflow-x: auto; }}
            .button {{ background: #4CAF50; border: none; color: white; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 4px; }}
        </style>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body>
        <div class="container">
            <h1>MyNet 模型分析</h1>
            
            <div class="card">
                <h2>模型概览</h2>
                <p><strong>模型名称:</strong> {model_info["model_name"]}</p>
                <p><strong>棋盘大小:</strong> {model_info["board_size"]}</p>
                <p><strong>节点数量:</strong> {model_info["nodes_count"]}</p>
                <p><strong>输入:</strong></p>
                <pre>{json.dumps(model_info["inputs"], indent=2)}</pre>
                <p><strong>输出:</strong></p>
                <pre>{json.dumps(model_info["outputs"], indent=2)}</pre>
                
                <a href="{onnx_path}" download class="button">下载ONNX模型</a>
                <a href="model_analysis.json" download class="button">下载模型分析JSON</a>
            </div>
            
            <div class="card">
                <h2>模型架构</h2>
                <p>MyNet是一个基于残差网络设计的深度神经网络模型，专为围棋游戏设计。它包含:</p>
                <ul>
                    <li>初始卷积层，将输入转换为特征图</li>
                    <li>3个残差块，每个残差块包含两个卷积层和批归一化</li>
                    <li>双头输出:
                        <ul>
                            <li>策略头: 预测下一步最佳落子位置的概率分布</li>
                            <li>价值头: 评估当前局面的胜率</li>
                        </ul>
                    </li>
                </ul>
            </div>
            
            <div class="card">
                <h2>操作类型分布</h2>
                <div class="chart">
                    <canvas id="operationsChart"></canvas>
                </div>
            </div>
            
            <div class="card">
                <h2>使用Netron查看交互式模型图</h2>
                <p>Netron是一个专门用于可视化神经网络模型的工具。使用以下命令查看:</p>
                <pre>pip install netron\npython -m netron {onnx_path}</pre>
                <p>或者访问 <a href="https://netron.app" target="_blank">https://netron.app</a> 并上传ONNX模型文件。</p>
            </div>
        </div>
        
        <script>
            // 绘制操作类型图表
            const ctx = document.getElementById('operationsChart').getContext('2d');
            const operationsChart = new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: {list(model_info["operations"].keys())},
                    datasets: [{{
                        label: '操作类型数量',
                        data: {list(model_info["operations"].values())},
                        backgroundColor: 'rgba(75, 192, 192, 0.6)'
                    }}]
                }},
                options: {{
                    scales: {{
                        y: {{
                            beginAtZero: true
                        }}
                    }}
                }}
            }});
        </script>
    </body>
    </html>
    """
    
    # 保存HTML文件
    html_path = viz_dir / "model_analysis.html"
    with open(html_path, "w") as f:
        f.write(html_content)
    
    print(f"可视化页面已创建: {html_path}")
    
    # 打开HTML页面
    if open_browser:
        webbrowser.open(f'file://{html_path.absolute()}')
        print("已在浏览器中打开可视化页面")
    
    # 使用netron可视化(可选)
    use_netron = True
    if open_browser and use_netron:
        print("正在启动Netron可视化服务...")
        netron.start(str(onnx_path))
    else:
        print("\n要使用Netron查看交互式模型结构，请运行:")
        print(f"python -m netron {onnx_path}")

if __name__ == "__main__":
    visualize_mynet_with_onnx() 