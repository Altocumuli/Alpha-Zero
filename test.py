import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"PyTorch中的CUDA版本: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    
    # 进行简单的CUDA操作测试
    x = torch.tensor([1.0, 2.0, 3.0]).cuda()
    y = x * 2
    print(f"CUDA张量测试: {x} * 2 = {y}")