# Megatron源码阅读
![image](https://github.com/user-attachments/assets/7a66d2d5-94a5-4587-a021-dd4905b671c5)

做数据并行，需要一个这种ddp或者dp先把模型包一下，处理模型副本的拷贝
ddp本身至少应该包含：规定怎么在训练开始前分发模型，怎么在训练结束后规约梯度。

## torchDDP -- 数据并行 -- DP和DDP
在PyTorch中，实现数据并行的主要方法是使用`torch.nn.DataParallel`和`torch.nn.parallel.DistributedDataParallel`（DDP）。以下是这两种方法的简要概述：

### 1. DataParallel(DP: 单机多卡, 单进程多线程，效率很低)
`DataParallel`是PyTorch中用于单机多GPU数据并行的类。它通过复制模型到每个GPU上，并将数据分割后发送到各个GPU上进行并行计算。

**基本写法**:
![0c724ae5fe03f9060c3596a9ef36e26](https://github.com/user-attachments/assets/f61067d3-64d6-48bc-9660-1fd269c73e42)

```python
import torch.nn as nn

model = nn.Linear(10, 10)  # 假设有一个模型
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

**使用`device_ids`指定GPU**:
```python
model = nn.DataParallel(model, device_ids=[0, 1])
```

### 2. DistributedDataParallel(DDP: 在每个rank上单独启进程，真正的多进程分布式)
`DistributedDataParallel`是用于多机多GPU数据并行的类。它需要配合`torch.distributed`通信包使用。

**基本写法**:
```python
from torch.nn.parallel import DistributedDataParallel as DDP

model = model.to(rank)  # 将模型移到对应的GPU
ddp_model = DDP(model, device_ids=[rank])
```

**使用`DistributedSampler`**:
```python
from torch.utils.data.distributed import DistributedSampler

sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
data_loader = DataLoader(dataset, batch_size=32, sampler=sampler)

## LocalDDP -- 数据并行
