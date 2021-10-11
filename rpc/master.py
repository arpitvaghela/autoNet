import os
import train

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'

import torch as th
import torch.distributed.rpc as rpc

rpc.init_rpc("master", rank=0, world_size=2)
args = (th.Tensor([1,1]), th.Tensor([2,2]))
ret = rpc.rpc_sync("worker0", train.train)
print(ret)
rpc.shutdown()
