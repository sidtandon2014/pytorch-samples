# References
# https://discuss.pytorch.org/t/distributedsampler/90205
# https://github.com/pytorch/examples/tree/main/distributed/ddp
# https://medium.com/codex/a-comprehensive-tutorial-to-pytorch-distributeddataparallel-1f4b42bb1b51
# https://github.com/pytorch/pytorch/blob/main/torch/utils/data/distributed.py#L68
import torch.distributed as dist
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import numpy as np
import torch.multiprocessing as mp

torch.cuda.is_available()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
def cleanup():
    dist.destroy_process_group()


def prepare(rank, world_size, batch_size=2, pin_memory=False, num_workers=0):
    dataset = TensorDataset(torch.tensor(np.array([[1.,2.,3.,4.,5.,6.,7.,8.,9.,0.]]).reshape(-1,1), dtype = torch.float32), torch.randn((10,1)))
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers
                            , drop_last=False, shuffle=False, sampler=sampler)
    
    return dataloader

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(1, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 1)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def ddp_main(rank, world_size):
    # setup the process groups
    setup(rank, world_size)
    # prepare the dataloader
    dataloader = prepare(rank, world_size)
    
    # instantiate the model(it's your own model) and move it to the right device\
    # torch.cuda.set_device(rank)
    model = ToyModel().to(rank)
    
    # wrap the model with DDP
    # device_ids tell DDP where is your model
    # output_device tells DDP where to output, in our case, it is rank
    # find_unused_parameters=True instructs DDP to find unused output of the forward() function of any module in the model
    ddp_model  = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    #################### The above is defined previously
   
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    for epoch in range(5):
        # if we are using DistributedSampler, we have to tell it which epoch this is
        dataloader.sampler.set_epoch(epoch)       
        print(f"----------------Epoch: {epoch}------------")
        for step, (x, y) in enumerate(dataloader):
            x = x.to(rank)
            y = y.to(rank)
            print(f"Rank: {rank}, data: {x}")
            optimizer.zero_grad(set_to_none=True)
            
            pred = ddp_model(x)
            label = y # x['label']
            
            loss = loss_fn(pred, label)
            loss.backward()
            optimizer.step()
    cleanup()


if __name__ == '__main__':
    # suppose we have 3 gpus
    world_size = 2    
    mp.spawn(
        ddp_main,
        args=(world_size,),
        nprocs=world_size
    )

