import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from functools import reduce

from env.base_env import BaseGame

class BaseNetConfig:
    def __init__(
        self, 
        num_channels:int = 256,
        dropout:float = 0.3,
        linear_hidden:list[int] = [256, 128],
    ):
        self.num_channels = num_channels
        self.linear_hidden = linear_hidden
        self.dropout = dropout
        
class MLPNet(nn.Module):
    def __init__(self, observation_size:tuple[int, int], action_space_size:int, config:BaseNetConfig, device:torch.device='cpu'):
        super().__init__()
        self.config = config
        self.device = device
        input_dim = observation_size[0] * observation_size[1] if len(observation_size) == 2 else observation_size[0]
        self.layer1 = nn.Linear(input_dim, config.linear_hidden[0])
        self.layer2 = nn.Linear(config.linear_hidden[0], config.linear_hidden[1])
        
        self.policy_head = nn.Linear(config.linear_hidden[1], action_space_size)
        self.value_head = nn.Linear(config.linear_hidden[1], 1)
        self.relu = nn.ReLU()
        self.to(device)

    def forward(self, x: torch.Tensor):
        #                                                         x: batch_size x board_x x board_y
        x = x.view(x.size(0), -1) # reshape tensor to 1d vectors, x.size(0) is batch size
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        pi = self.policy_head(x)
        v = self.value_head(x)
        return F.log_softmax(pi, dim=1), torch.tanh(v)


class LinearModel(nn.Module):
    def __init__(self, observation_size:tuple[int, int], action_space_size:int, config:BaseNetConfig, device:torch.device='cpu'):
        super(LinearModel, self).__init__()
        
        self.action_size = action_space_size
        self.config = config
        self.device = device
        
        observation_size = reduce(lambda x, y: x*y , observation_size, 1)
        self.l_pi = nn.Linear(observation_size, action_space_size)
        self.l_v  = nn.Linear(observation_size, 1)
        self.to(device)
    
    def forward(self, s: torch.Tensor):
        #                                                           s: batch_size x board_x x board_y
        s = s.view(s.shape[0], -1)                                # s: batch_size x (board_x * board_y)
        pi = self.l_pi(s)
        v = self.l_v(s)
        return F.log_softmax(pi, dim=1), torch.tanh(v)
    
class MyNet(nn.Module):
    def __init__(self, observation_size:tuple[int, int], action_space_size:int, config:BaseNetConfig, device:torch.device='cpu'):
        super().__init__()
        self.config = config
        self.device = device
        ########################
        # TODO: your code here #
        ########################
        # 确定输入维度
        if len(observation_size) == 2:
            self.board_size = observation_size
            input_channels = 1
        else:
            self.board_size = (int(np.sqrt(observation_size[0])), int(np.sqrt(observation_size[0])))
            input_channels = 1
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(input_channels, config.num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(config.num_channels)

        # 残差块
        self.res_blocks = nn.ModuleList([
            self._build_residual_block(config.num_channels)
            for _ in range(3)  # 使用3个残差块
        ])

        # 策略头
        self.policy_conv = nn.Conv2d(config.num_channels, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * self.board_size[0] * self.board_size[1], action_space_size)

        # 价值头
        self.value_conv = nn.Conv2d(config.num_channels, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * self.board_size[0] * self.board_size[1], config.linear_hidden[0])
        self.value_fc2 = nn.Linear(config.linear_hidden[0], 1)

        # Dropout层
        self.dropout = nn.Dropout(config.dropout)

        self.to(device)
    
    def _build_residual_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
        
    def forward(self, s: torch.Tensor):
        ########################
        # TODO: your code here #
        # 处理输入维度
        x = s
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # 添加批次维度

        if len(x.shape) == 3:
            x = x.view(x.size(0), 1, self.board_size[0], self.board_size[1])

        # 初始卷积层
        x = F.relu(self.bn1(self.conv1(x)))

        # 残差块
        for block in self.res_blocks:
            residual = x
            out = block(x)
            x = F.relu(out + residual)  # 添加残差连接

        # 策略头
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)

        # 价值头
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = self.dropout(value)
        value = self.value_fc2(value)

        return F.log_softmax(policy, dim=1), torch.tanh(value)
        
        ########################