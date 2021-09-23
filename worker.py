import os
import torch.distributed.rpc as rpc
import train

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'

rpc.init_rpc("worker0", rank=1, world_size=2)
print("Worker Started")
rpc.shutdown() 