# Megatron源码阅读
![image](https://github.com/user-attachments/assets/7a66d2d5-94a5-4587-a021-dd4905b671c5)

做数据并行，需要一个这种ddp或者dp先把模型包一下，处理模型副本的拷贝
ddp本身至少应该包含：规定怎么在训练开始前分发模型，怎么在训练结束后规约梯度。

## torchDDP -- 数据并行 -- DP和DDP
在PyTorch中，实现数据并行的主要方法是使用`torch.nn.DataParallel`和`torch.nn.parallel.DistributedDataParallel`（DDP）。以下是这两种方法的简要概述：

torch distributed overview
https://pytorch.org/tutorials/beginner/dist_overview.html

### 1. DataParallel(DP: 单机多卡, 单进程多线程，效率很低)
`DataParallel`是PyTorch中用于单机多GPU数据并行的类。它通过复制模型到每个GPU上，并将数据分割后发送到各个GPU上进行并行计算。

**基本写法**:

```python
import torch.nn as nn

model = nn.Linear(10, 10)  # 假设有一个模型
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model, device_ids=[0, 1])
```

**使用`device_ids`指定GPU**:
```python
model = nn.DataParallel(model, device_ids=[0, 1])
```
https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html#dataparallel

### 2. DistributedDataParallel(DDP: 在每个rank上单独启进程，真正的多进程分布式)
`DistributedDataParallel`是用于多机多GPU数据并行的类。它需要配合`torch.distributed`通信包使用。

DDP容器：DDP通过在每个模型副本之间同步梯度来提供数据并行性。
![image](https://github.com/user-attachments/assets/25249b2d-3f46-4824-ab74-710f84cb11bf)
**基本写法**:
```python
from torch.nn.parallel import DistributedDataParallel as DDP

model = model.to(rank)  # 将模型移到对应的GPU
ddp_model = DDP(model, device_ids=[rank])
```
https://pytorch.org/docs/stable/notes/ddp.html#distributeddataparallel


数据分片：DDP不会自动将输入数据分割或分片到参与的GPU上；用户需要自己定义如何进行数据分片，例如通过使用DistributedSampler。

**使用`DistributedSampler`**:
```python
from torch.utils.data.distributed import DistributedSampler

sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
data_loader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel

完整例子test_mul_node.py：
```python
        dist.init_process_group(backend='mpi')
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        def partition_dataset():
                train_sample = DistributedSampler(dataset)
                train_set = DataLoader(dataset,batch_size=bsz,shuffle=(train_sample is
                                     None),sampler=train_sample)
        def run(rank,size):
                model = MPI_Lenet()
                model = DistributedDataParallel(model)
                model.train()
```
0. 命令行启动分布式训练脚本时候，使用torch.distributed.launch 或者torchrun来启动单机多卡进程，负责传一些参数 比如用多少节点，多机情况下，要每个机器上都跑一个脚本，通过设置 master_addr 来告诉多机该找哪个机器通信，神威应该在启动的时候 会不太一样，走exec。
```python
python -m torch.distributed.launch $DISTRIBUTED_ARGS \
exec  python3 ./pretrain.py \
```
https://pytorch.org/docs/stable/elastic/run.html

1. DistributedSampler 划分数据集，确定每个数据并行节点所需要处理的数据。
```python
class DistributedSampler(Sampler[T_co]):
    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)
```
3. Dataloader 会根据数据并行rank来从数据集中抽取返回不同的数据

5. DistributedDataParallel  用来实现 每个数据并行节点在梯度计算后的规约操作。
```python

```
## LocalDDP -- 数据并行
