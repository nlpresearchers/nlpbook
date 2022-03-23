import torch

random_seed = 123
torch.manual_seed(random_seed)
print(torch.rand(1)) # 随机生成[0, 1)的数
print(torch.rand(1))