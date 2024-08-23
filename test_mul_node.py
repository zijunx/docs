import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets,transforms
from math import ceil
from torch.autograd import Variable
import os
import gzip
import codecs
import numpy as np
import sys
gbatch_size = 128
class MPI_Lenet(nn.Module):
        def __init__(self):
                super(MPI_Lenet,self).__init__()
                self.convnet = nn.Sequential(
                        nn.Conv2d(1,10,kernel_size=5),
                        nn.MaxPool2d(kernel_size=2),
                        nn.ReLU(),
                        nn.Conv2d(10,20,kernel_size=5),
                        nn.MaxPool2d(kernel_size=2),
                        nn.ReLU()
                        )
                self.feedforward = nn.Sequential(
                        nn.Linear(320,50),
                        nn.ReLU(),
                        nn.Dropout(p=0.5),
                        nn.Linear(50,10)
                        )
                for p in self.convnet.parameters():
                        if len(p.shape)>1: nn.init.xavier_normal_(p)
                        for p in self.feedforward.parameters():
                                if len(p.shape)>1: nn.init.xavier_normal_(p)
        def forward(self,x):
                return self.feedforward(self.convnet(x).view(-1,320))
def get_int(b):
                return int(codecs.encode(b, 'hex'), 16)
def partition_dataset():
                dataset = datasets.MNIST('$user_dir/data', train=True, download=False,
                                transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                                                                ]))
                size = dist.get_world_size()
                bsz = int(gbatch_size/size)
                train_sample = DistributedSampler(dataset)
                train_set = DataLoader(dataset,batch_size=bsz,shuffle=(train_sample is
                                     None),sampler=train_sample)
                return train_set,bsz
def run(rank,size):
                torch.manual_seed(1)
                train_set , bsz = partition_dataset()
                model = MPI_Lenet()
                model = DistributedDataParallel(model)
                model.train()
                optimizer = torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.5)
                num_batches = ceil(len(train_set.dataset))
                for epoch in range(10):
                        for batch_id,(data,target) in enumerate(train_set):
                                data,target = Variable(data),Variable(target)
                                optimizer.zero_grad()
                                output = model(data)
                                loss_fct = torch.nn.CrossEntropyLoss()
                                loss=loss_fct(output,target)
#loss = torch.nn.functional.nll_loss(output,target)
                                loss.backward()
                                optimizer.step()
                                if rank==0:
                                        print('Batch_is {},Epoch {} Loss {:.6f} Global batch size {} on{} ranks'.format(batch_id,epoch,loss,gbatch_size,dist.get_world_size()))
#sys.stdout.write(loss.data)
def init_print(rank,size,debug_print=True):
                if not debug_print:
                        if rank>0:
                                sys.stdout=open(os.devnull,'w')
                                sys.stderr=open(os.devnull,'w')
                else:
                        old_out = sys.stdout
                        class LabeledStdout:
                                def __init__(self,rank,size):
                                        self._r = rank
                                        self._s = size
                                        self.flush = sys.stdout.flush
                                def write(self,x):
                                        if x=='\n':
                                                old_out.write(x)
                                        else:
                                                old_out.write('[%d,%d]%s'%(self._r,self._s,x))
                                        sys.stdout = LabeledStdout(rank,size)
if __name__=="__main__":
        dist.init_process_group(backend='mpi')
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        init_print(rank,world_size)
        if rank==0:
                print('num of process is %d'%world_size)
        run(dist.get_rank(),dist.get_world_size())

